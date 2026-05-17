import os


def _next_exp_folder(base_folder):
    exp_folder = os.path.join(base_folder, "exp")
    if not os.path.exists(exp_folder):
        return exp_folder

    exp_num = 1
    while True:
        candidate = os.path.join(base_folder, f"exp{exp_num}")
        if not os.path.exists(candidate):
            return candidate
        exp_num += 1


def create_exp_folder(model_name="default"):
    if not os.path.exists("run"):
        os.mkdir("run")

    train_folder = os.path.join("run", "train")
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    model_folder = os.path.join(train_folder, model_name)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    exp_folder = _next_exp_folder(model_folder)
    os.mkdir(exp_folder)
    weights_folder = os.path.join(exp_folder, f"weights")
    os.mkdir(weights_folder)
    return exp_folder, weights_folder


def create_val_exp_folder(model_name="default"):
    if not os.path.exists("run"):
        os.mkdir("run")

    predict_folder = os.path.join("run", "predict")
    if not os.path.exists(predict_folder):
        os.mkdir(predict_folder)

    model_folder = os.path.join(predict_folder, model_name)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    exp_folder = _next_exp_folder(model_folder)
    os.mkdir(exp_folder)
    return exp_folder
