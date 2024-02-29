import copy
from ikomia import core, dataprocess, utils
import sys
import os.path as osp
import torch
from torchvision.ops import nms
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmyolo.registry import RUNNERS
from infer_yolo_world.download_yolo_world_weights import download_model_weights

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloWorldParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = 'yolo_world_m'
        self.cuda = torch.cuda.is_available()
        self.prompt = 'person, dog, cat'
        self.max_dets = 100
        self.conf_thres = 0.1
        self.iou_thres = 0.25
        self.use_custom_model = False
        self.config_file = ""
        self.model_weight_file = ""
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = str(params["model_name"])
        self.cuda = utils.strtobool(params["cuda"])
        self.prompt = str(params["prompt"])
        self.max_dets = int(params["max_dets"])
        self.conf_thres = float(params["conf_thres"])
        self.iou_thres = float(params["iou_thres"])
        self.config_file = params["config_file"]
        self.use_custom_model = utils.strtobool(params["use_custom_model"])
        self.model_weight_file = params["model_weight_file"]
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["model_name"] = str(self.model_name)
        params["cuda"] = str(self.cuda)
        params["prompt"] = str(self.prompt)
        params["max_dets"] = str(self.max_dets)
        params["conf_thres"] = str(self.conf_thres)
        params["iou_thres"] = str(self.iou_thres)
        params["config_file"] = str(self.config_file)
        params["model_weight_file"] = str(self.model_weight_file)
        params["use_custom_model"] = str(self.use_custom_model)

        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloWorld(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Create parameters object
        if param is None:
            self.set_param_object(InferYoloWorldParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.runner = None
        self.config_path_base = osp.join(osp.dirname(osp.realpath(__file__)),
                                                            "YOLO-World", "configs", "pretrain")

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def parse_cfg_options(self, cfg_options):
        parsed_options = {}
        for key, value in cfg_options.items():
            try:
                parsed_value = eval(value)
            except:
                # If eval fails or is not safe, use the value directly.
                parsed_value = value
            parsed_options[key] = parsed_value
        return parsed_options

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Get input
        input = self.get_input(0)

        if param.update or self.runner is None:
            self.device = torch.device(
                "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            # Get model and config file
            if param.model_weight_file != "":
                model_weights = param.model_weight_file
                if param.config_file != "":
                    self.config_file = param.config_file
                else:
                    print('Error: Please provide a path to the model config file.')
                    sys.exit(1)
            else:
                model_weights, config_name = download_model_weights(param.model_name)

            # Load config
            config_file = osp.join(self.config_path_base, config_name)
            cfg = Config.fromfile(config_file)
            cfg.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(config_file))[0])
            cfg.load_from = model_weights

            # Load model
            if 'runner_type' not in cfg:
                self.runner = Runner.from_cfg(cfg)
            else:
                self.runner = RUNNERS.build(cfg)

            self.runner.call_hook('before_run')
            self.runner.load_or_resume()
            pipeline = cfg.test_dataloader.dataset.pipeline
            self.runner.pipeline = Compose(pipeline)
            self.runner.model.eval().to(self.device)

        # Get and set classes
        texts = [[t.strip()] for t in param.prompt.split(',')] + [[' ']]
        classes = [item[0] for item in texts if item[0].strip()]
        self.set_names(classes)

        # Data input pre-process
        data_info = dict(img_id=0, img_path=input.source_file_path, texts=texts)
        data_info = self.runner.pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                        data_samples=[data_info['data_samples']])
    
        # Inference
        with torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
             
        keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=param.iou_thres)
        pred_instances = pred_instances[keep]
        pred_instances = pred_instances[pred_instances.scores.float() > param.conf_thres]  
        
        if len(pred_instances.scores) > param.max_dets:
            indices = pred_instances.scores.float().topk(param.max_dets)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()

        # Display output
        for i, (box, conf, cls) in enumerate(zip(
                                    pred_instances['bboxes'],
                                    pred_instances['scores'],
                                    pred_instances['labels'])
                                ):
            x1, y1 = box[0], box[1]
            widht = box[2] - x1
            height = box[3] - y1
            self.add_object(
                i,
                int(cls),
                float(conf),
                float(x1),
                float(y1),
                float(widht),
                float(height)
            )

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloWorldFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_yolo_world"
        self.info.short_description = "YOLO-World is a real-time zero-shot object detection model" \
                                    "that leverages the power of open-vocabulary learning to " \
                                    "recognize and localize a wide range of objects in images."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/logo.png"
        self.info.authors = "Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying"
        self.info.article = "YOLO-World: Real-Time Open-Vocabulary Object Detection"
        self.info.journal = "arXiv:2401.17270"
        self.info.year = 2024
        self.info.license = "GNU General Public License v3.0"
        # URL of documentation
        self.info.documentation_link = "https://mmdetection.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolo_world"
        self.info.original_repository = "https://github.com/AILab-CVC/YOLO-World"
        # Keywords used for search
        self.info.keywords = "YOLO, zero-shot, mmdet, mmlab, mmyolo, Tencent  AI, PyTorch"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return InferYoloWorld(self.info.name, param)
