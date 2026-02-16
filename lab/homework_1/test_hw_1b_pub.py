import pandas as pd
from lab_2_hfl.hfl_complete import *

seed = 10
split_one = split(100, True, seed)


def test_fedsgd_weight_one(tb):
    src = tb.cells[tb._cell_index("fedsgd_weight")].source
    src_symbs = {}
    exec(src, globals().copy(), src_symbs)

    lr_ = 0.1
    c_ = 0.2
    nr_rounds_ = 5

    fedsgd_gradient_server = FedSgdServer(lr_, split_one, c_, seed)
    fedsgd_gradient_res = fedsgd_gradient_server.run(nr_rounds_)

    fedsgd_weight_server = src_symbs["FedSgdWeightServer"](
        lr_, split_one, c_, seed)
    fedsgd_weight_res = fedsgd_weight_server.run(nr_rounds_)

    diff_acc = np.array(fedsgd_gradient_res.test_accuracy) - \
        np.array(fedsgd_weight_res.test_accuracy)

    assert (np.abs(diff_acc) < 0.1).all().item()

    assert fedsgd_gradient_res.message_count == fedsgd_weight_res.message_count


def test_fedavg_grad_one(tb):
    src = tb.cells[tb._cell_index("fedavg_grad")].source
    src_symbs = {}
    exec(src, globals().copy(), src_symbs)

    lr_ = 0.1
    bs_ = 100
    c_ = 0.2
    e_ = 2
    nr_rounds_ = 5

    fedavg_weight_server = FedAvgServer(
        lr_, bs_, split_one, c_, e_, seed)
    fedavg_weight_res = fedavg_weight_server.run(nr_rounds_)

    fedavg_gradient_server = src_symbs["FedAvgGradServer"](
        lr_, bs_, split_one, c_, e_, seed)
    fedavg_gradient_res = fedavg_gradient_server.run(nr_rounds_)

    diff_acc = np.array(fedavg_weight_res.test_accuracy) - \
        np.array(fedavg_gradient_res.test_accuracy)

    assert (np.abs(diff_acc) < 0.2).all().item()

    assert fedavg_weight_res.message_count == fedavg_gradient_res.message_count


def test_client_number_experiments(tb):
    src = tb.cells[tb._cell_index("client_number_experiments")].source
    src_symbs = {}
    exec(src, globals().copy(), src_symbs)

    ref_df = pd.DataFrame({
        "Algorithm": ["FedSGD", "FedAvg"] * 3,
        "N": [10, 10, 50, 50, 100, 100],
        "C": [0.1] * 6,
        "Message count": [0] * 6,
        "Test accuracy": [0.] * 6
    })

    ret_df = src_symbs["client_number_experiments"]()

    ref_df["Message count"] = ret_df["Message count"]
    ref_df["Test accuracy"] = ret_df["Test accuracy"]

    assert ref_df.equals(ret_df)
    assert (ret_df["Message count"] > 0).all().item()
    assert (ret_df["Test accuracy"] > 0.).all().item()


def test_client_fraction_experiments(tb):
    src = tb.cells[tb._cell_index("client_fraction_experiments")].source
    src_symbs = {}
    exec(src, globals().copy(), src_symbs)

    ref_df = pd.DataFrame({
        "Algorithm": ["FedSGD", "FedAvg"] * 3,
        "N": [100] * 6,
        "C": [0.01, 0.01, 0.1, 0.1, 0.2, 0.2],
        "Message count": [0] * 6,
        "Test accuracy": [0.] * 6
    })

    ret_df = src_symbs["client_fraction_experiments"]()

    ref_df["Message count"] = ret_df["Message count"]
    ref_df["Test accuracy"] = ret_df["Test accuracy"]

    assert ref_df.equals(ret_df)
    assert (ret_df["Message count"] > 0).all().item()
    assert (ret_df["Test accuracy"] > 0.).all().item()


def test_local_epoch_experiments(tb):
    src = tb.cells[tb._cell_index("local_epoch_experiments")].source
    src_symbs = {}
    exec(src, globals().copy(), src_symbs)

    ret_dict = src_symbs["local_epoch_experiments"]()

    assert isinstance(ret_dict, dict)
    assert len(ret_dict) == 4

    arr_fedsgd_1 = np.array(ret_dict["FedSGD E=1"])
    arr_fedavg_1 = np.array(ret_dict["FedAvg E=1"])
    arr_fedavg_2 = np.array(ret_dict["FedAvg E=2"])
    arr_fedavg_4 = np.array(ret_dict["FedAvg E=4"])

    assert (arr_fedsgd_1 < arr_fedavg_1).all().item()
    assert (arr_fedavg_1 < arr_fedavg_2).all().item()
    assert (arr_fedavg_2 < arr_fedavg_4).all().item()
