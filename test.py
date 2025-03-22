import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        # DAT baseline, ICCV 2023
        from models.team00_DAT import main as DAT
        name = f"{model_id:02}_DAT_baseline"
        model_path = os.path.join('model_zoo', 'team00_dat.pth')
        model_func = DAT
        
    elif model_id == 25:
        # define your model and load the checkpoint
        from models.team25_HAT import main as HAT
        name = f"{model_id:02}_HAT"
        model_path = os.path.join('model_zoo/team25_HAT', 'team25_HAT.pth')
        model_func = HAT
        
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    return model_func, model_path, name


def run(model_func, model_name, model_path, device, args, mode="test"):
    # --------------------------------
    # dataset path
    # --------------------------------
    if mode == "valid":
        data_path = args.valid_dir
    elif mode == "test":
        data_path = args.test_dir
    assert data_path is not None, "Please specify the dataset path for validation or test."
    
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    model_func(model_dir=model_path, input_path=data_path, output_path=save_path, device=device)
    end.record()
    torch.cuda.synchronize()
    print(f"Model {model_name} runtime (Including I/O): {start.elapsed_time(end)} ms")


def main(args):
    utils_logger.logger_info("NTIRE2025-ImageSRx4", log_path="NTIRE2025-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2025-ImageSRx4")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model_func, model_path, model_name = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if args.valid_dir is not None:
        print('run valid')
        run(model_func, model_name, model_path, device, args, mode="valid")
        
    if args.test_dir is not None:
        print('run test')
        run(model_func, model_name, model_path, device, args, mode="test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2025-ImageSRx4")
    parser.add_argument("--valid_dir", default='/work/u1657859/ming0531/NTIRE2025_ImageSR_x4/data/DIV2K_valid_LR_bicubic/X4/', type=str, help="Path to the validation set")
    parser.add_argument("--test_dir", default='/work/u1657859/ming0531/NTIRE2025_ImageSR_x4/data/DIV2K_test_LR_bicubic/X4/', type=str, help="Path to the test set")
    parser.add_argument("--save_dir", default="results", type=str)
    parser.add_argument("--model_id", default=25, type=int)

    args = parser.parse_args()
    pprint(args)

    main(args)
