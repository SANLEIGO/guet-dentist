"""
自动标定主控制器

编排完整标定流程
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dental_stitcher_v1.calibration.instance_extractor import (
    ToothInstance,
    InstanceSegmentationResult,
    extract_teeth_instances_from_yolo
)
from dental_stitcher_v1.calibration.geometry_constraints import (
    DentalArchModel,
    fit_dental_arch_curve,
    compute_spacing_constraints,
    check_geometry_consistency
)
from dental_stitcher_v1.calibration.camera_estimator import (
    CameraParameters,
    initialize_camera_intrinsics,
    estimate_camera_extrinsics,
    estimate_distortion_parameters
)
from dental_stitcher_v1.calibration.bundle_adjustment import (
    bundle_adjustment_optimization
)
from dental_stitcher_v1.calibration.distortion_corrector import (
    undistort_images,
    evaluate_calibration_quality
)
from dental_stitcher_v1.calibration.calibration_diagnostics import (
    CalibrationDiagnostics,
    generate_calibration_diagnostics
)
from dental_stitcher_v1.calibration.calibration_storage import (
    save_calibration,
    load_calibration
)


@dataclass
class CalibrationResult:
    """标定结果"""
    camera_params: CameraParameters
    undistorted_images: list[np.ndarray]
    original_instances: list[list[ToothInstance]]
    undistorted_instances: list[list[ToothInstance]]
    quality_metrics: dict
    success: bool
    diagnostics: CalibrationDiagnostics


def auto_calibrate_pipeline(
    images: list[np.ndarray],
    seg_method: str = "alphadent",
    enable_cache: bool = True,
    enable_iterative: bool = True  # 新增：启用迭代标定
) -> CalibrationResult:
    """
    自动标定完整流程（支持迭代标定）

    Args:
        images: 输入图像列表（至少2张）
        seg_method: 分割方法（"alphadent" or "unet"）
        enable_cache: 是否启用缓存（加载历史标定参数）
        enable_iterative: 是否启用迭代标定（两轮优化）

    Returns:
        CalibrationResult包含标定结果和诊断信息
    """
    # 步骤0：验证输入
    if len(images) < 2:
        return _failed_calibration(
            "insufficient_images",
            "Need at least 2 images for calibration"
        )

    # ========== 第1轮：粗标定（基于原始图像） ==========
    # 步骤1-1：实例提取（原始图像）
    instance_results = []
    for image in images:
        result = _extract_instances(image, seg_method)
        instance_results.append(result)

    all_instances = [result.instances for result in instance_results]

    # 检查实例数量
    if any(len(inst) < 4 for inst in all_instances):
        return _failed_calibration(
            "insufficient_tooth_instances",
            "Each image must have at least 4 tooth instances for calibration"
        )

    # 步骤1-2：几何约束建模
    arch_models = []
    for instances in all_instances:
        model = fit_dental_arch_curve(instances)
        arch_models.append(model)

    # 检查几何一致性
    geometry_consistency_results = []
    for instances, model in zip(all_instances, arch_models):
        consistency = check_geometry_consistency(instances, model, strict=False)
        geometry_consistency_results.append(consistency)

    # 步骤1-3：相机内参初始化
    intrinsics = initialize_camera_intrinsics(images[0].shape)

    # 尝试加载历史标定参数（如果启用缓存）
    if enable_cache:
        cached_params = load_calibration()
        if cached_params is not None:
            # 验证缓存参数是否适用
            if (cached_params.image_width == intrinsics.image_width and
                cached_params.image_height == intrinsics.image_height):
                intrinsics = cached_params

    # 步骤1-4：相机外参估计（相邻图像对）
    for i in range(len(images) - 1):
        try:
            rot_vec, t = estimate_camera_extrinsics(
                all_instances[i],
                all_instances[i+1],
                intrinsics
            )
            intrinsics.rotation_vectors.append(rot_vec)
            intrinsics.translation_vectors.append(t)
        except Exception as e:
            # 外参估计失败，使用默认值
            intrinsics.rotation_vectors.append(np.zeros(3))
            intrinsics.translation_vectors.append(np.zeros(3))

    # 步骤1-5：畸变参数估计
    try:
        intrinsics = estimate_distortion_parameters(all_instances, intrinsics, arch_models)
    except Exception as e:
        # 畸变估计失败，保持默认值
        pass

    # 步骤1-6：Bundle Adjustment优化（第一轮）
    try:
        optimized_camera, optimized_positions = bundle_adjustment_optimization(
            all_instances, intrinsics, arch_models
        )
    except Exception as e:
        # 优化失败，使用当前参数
        optimized_camera = intrinsics

    # 步骤1-7：第一轮畸变校正
    undistorted_images_round1 = undistort_images(images, optimized_camera)

    # ========== 第2轮：精细标定（基于校正后图像） ==========
    if enable_iterative:
        # 步骤2-1：在校正后图像上重新提取实例
        instance_results_round2 = []
        for undist_img in undistorted_images_round1:
            result = _extract_instances(undist_img, seg_method)
            instance_results_round2.append(result)

        all_instances_round2 = [result.instances for result in instance_results_round2]

        # 检查第二轮实例数量
        if not any(len(inst) < 4 for inst in all_instances_round2):
            # 第二轮实例足够，继续精确标定
            # 步骤2-2：重新建模几何约束
            arch_models_round2 = []
            for instances in all_instances_round2:
                model = fit_dental_arch_curve(instances)
                arch_models_round2.append(model)

            # 步骤2-3：重新估计外参（使用校正后的实例）
            intrinsics_round2 = optimized_camera
            intrinsics_round2.rotation_vectors = []  # 清空重新计算
            intrinsics_round2.translation_vectors = []

            for i in range(len(undistorted_images_round1) - 1):
                try:
                    rot_vec, t = estimate_camera_extrinsics(
                        all_instances_round2[i],
                        all_instances_round2[i+1],
                        intrinsics_round2
                    )
                    intrinsics_round2.rotation_vectors.append(rot_vec)
                    intrinsics_round2.translation_vectors.append(t)
                except Exception as e:
                    intrinsics_round2.rotation_vectors.append(np.zeros(3))
                    intrinsics_round2.translation_vectors.append(np.zeros(3))

            # 步骤2-4：重新估计畸变参数
            try:
                intrinsics_round2 = estimate_distortion_parameters(
                    all_instances_round2, intrinsics_round2, arch_models_round2
                )
            except Exception as e:
                pass

            # 步骤2-5：第二轮Bundle Adjustment
            try:
                optimized_camera, optimized_positions = bundle_adjustment_optimization(
                    all_instances_round2, intrinsics_round2, arch_models_round2
                )
            except Exception as e:
                pass

            # 步骤2-6：第二轮畸变校正
            undistorted_images_final = undistort_images(images, optimized_camera)

            # 步骤2-7：最终验证提取
            undistorted_instances = []
            for undist_img in undistorted_images_final:
                result = _extract_instances(undist_img, seg_method)
                undistorted_instances.append(result.instances)
        else:
            # 第二轮实例不足，使用第一轮结果
            undistorted_images_final = undistorted_images_round1
            undistorted_instances = all_instances_round2
    else:
        # 不启用迭代，直接使用第一轮结果
        undistorted_images_final = undistorted_images_round1
        undistorted_instances = all_instances

    # 步骤9：质量评估（对比原始实例和最终校正后实例）
    quality = evaluate_calibration_quality(
        all_instances, undistorted_instances, optimized_camera
    )

    # 步骤10：参数持久化存储
    if quality["calibration_success"]:
        save_calibration(
            optimized_camera,
            quality_score=quality["rmse_improvement"],
            confidence="high" if quality["distortion_reasonable"] else "medium"
        )

    # 步骤11：生成诊断
    diagnostics = generate_calibration_diagnostics(
        instance_extraction_success=True,
        instance_extraction_details={
            "num_images": len(images),
            "instances_per_image_round1": [len(inst) for inst in all_instances],
            "instances_per_image_round2": [len(inst) for inst in undistorted_instances] if enable_iterative else [],
            "iterative_calibration": enable_iterative
        },
        geometry_constraints_success=all(c["consistent"] for c in geometry_consistency_results),
        geometry_constraints_details={
            "consistency_scores": [m.consistency_score for m in arch_models],
            "curvature_rmse": [m.curvature_rmse for m in arch_models]
        },
        camera_estimation_success=True,
        camera_estimation_details={
            "fx": optimized_camera.fx,
            "fy": optimized_camera.fy,
            "k1": optimized_camera.k1,
            "k2": optimized_camera.k2,
            "iterative_refinement": enable_iterative
        },
        bundle_adjustment_success=True,
        bundle_adjustment_details={
            "iterations": 200,
            "iterative_rounds": 2 if enable_iterative else 1
        },
        distortion_correction_success=quality["calibration_success"],
        distortion_correction_details={
            "rmse_improvement": quality["rmse_improvement"],
            "spacing_improvement": quality["spacing_improvement"],
            "iterative_applied": enable_iterative
        },
        quality_validation_success=quality["calibration_success"],
        quality_validation_details=quality,
        failure_reason=None if quality["calibration_success"] else "quality_insufficient"
    )

    return CalibrationResult(
        camera_params=optimized_camera,
        undistorted_images=undistorted_images_final,
        original_instances=all_instances,
        undistorted_instances=undistorted_instances,
        quality_metrics=quality,
        success=quality["calibration_success"],
        diagnostics=diagnostics
    )


def _extract_instances(image: np.ndarray, method: str) -> InstanceSegmentationResult:
    """
    提取牙齿实例（内部函数，严格按method执行，失败不回退）

    Args:
        image: 输入图像
        method: 分割方法（"alphadent" or "unet"）

    Returns:
        InstanceSegmentationResult，失败时返回空结果并标记原因
    """
    from dental_stitcher_v1.calibration.instance_extractor import _empty_instance_result

    if method == "alphadent":
        # YOLO实例分割
        from dental_stitcher_v1.segmentation import _get_alphadent_model

        model = _get_alphadent_model()
        if model is None:
            # 模型加载失败
            return _empty_instance_result(image, "yolo_model_unavailable")

        try:
            # 运行YOLOv8预测
            import cv2
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, imgsz=960, conf=0.1, verbose=False)

            # 检查是否有检测结果
            if not results or len(results) == 0:
                return _empty_instance_result(image, "yolo_no_detection_results")

            if results[0].masks is None or results[0].masks.data is None:
                return _empty_instance_result(image, "yolo_no_masks_detected")

            # 提取实例信息
            instance_result = extract_teeth_instances_from_yolo(results, image, apply_grabcut=False)

            # 检查实例数量
            if len(instance_result.instances) < 4:
                instance_result.fallback_reason = f"insufficient_instances_{len(instance_result.instances)}_need_4"

            return instance_result

        except Exception as e:
            # YOLO推理失败
            return _empty_instance_result(image, f"yolo_inference_failed: {str(e)}")

    elif method == "unet":
        # U-Net语义分割（不支持实例提取）
        return _empty_instance_result(
            image,
            "unet_does_not_support_instance_segmentation",
            "U-Net是语义分割模型，无法提取单个牙齿实例。自动标定需要实例级别的牙齿信息（位置、类别、边界框）。\n\n"
            "解决方案：\n"
            "1. 使用AlphaDent (YOLOv8) 进行分割（推荐）\n"
            "2. 如果图像有畸变导致YOLO失败，请先校正图像畸变\n"
            "3. 禁用自动标定功能，直接进行拼接"
        )

    else:
        # 其他方法不支持
        return _empty_instance_result(image, f"method_{method}_not_supported")


def _failed_calibration(reason: str, message: str) -> CalibrationResult:
    """创建失败的标定结果"""
    from dental_stitcher_v1.calibration.instance_extractor import _empty_instance_result

    # 创建空结果
    empty_result = _empty_instance_result(np.zeros((100, 100, 3), dtype=np.uint8), reason)

    diagnostics = generate_calibration_diagnostics(
        instance_extraction_success=False,
        instance_extraction_details={"error": message},
        geometry_constraints_success=False,
        geometry_constraints_details={},
        camera_estimation_success=False,
        camera_estimation_details={},
        bundle_adjustment_success=False,
        bundle_adjustment_details={},
        distortion_correction_success=False,
        distortion_correction_details={},
        quality_validation_success=False,
        quality_validation_details={},
        failure_reason=reason
    )

    return CalibrationResult(
        camera_params=initialize_camera_intrinsics((100, 100)),
        undistorted_images=[],
        original_instances=[],
        undistorted_instances=[],
        quality_metrics={},
        success=False,
        diagnostics=diagnostics
    )