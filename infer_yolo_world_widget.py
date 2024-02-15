from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_yolo_world.infer_yolo_world_process import InferYoloWorldParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferYoloWorldWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferYoloWorldParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model name
        self.combo_model = pyqtutils.append_combo(
            self.grid_layout, "Model name")
        self.combo_model.addItem("yolo_world_s")
        self.combo_model.addItem("yolo_world_m")
        self.combo_model.addItem("yolo_world_l")
        self.combo_model.addItem("yolo_world_l_plus")

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(
            self.grid_layout, "Prompt", self.parameters.prompt)

        # Confidence thresholds
        self.spin_conf_thres = pyqtutils.append_double_spin(
                                                self.grid_layout,
                                                "Confidence thresh",
                                                self.parameters.conf_thres,
                                                min=0., max=1., step=0.01, decimals=2)

        # Top k
        self.spin_top_k = pyqtutils.append_spin(
                                        self.grid_layout,
                                        "Top k",
                                        self.parameters.top_k,
                                        min=1
                                    )
        # Costum model
        self.check_custom_model = pyqtutils.append_check(
                                                self.grid_layout,
                                                "Use custom model",
                                                self.parameters.use_custom_model
                                            )

        self.browse_custom_cfg = pyqtutils.append_browse_file(
                                                self.grid_layout,
                                                "Custom config (.py)",
                                                self.parameters.config_file
                                                )
        self.browse_custom_weights = pyqtutils.append_browse_file(
                                                        self.grid_layout,
                                                        "Custom weights (.pth)",
                                                        self.parameters.model_weight_file
                                                )

        enabled = self.check_custom_model.isChecked()
        self.combo_model.setEnabled(not enabled)
        self.browse_custom_cfg.setEnabled(enabled)
        self.browse_custom_weights.setEnabled(enabled)

        self.check_custom_model.stateChanged.connect(self.on_check_custom_changed)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_check_custom_changed(self, b):
        enabled = self.check_custom_model.isChecked()
        self.combo_model.setEnabled(not enabled)
        self.browse_custom_cfg.setEnabled(enabled)
        self.browse_custom_weights.setEnabled(enabled)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.max_dets = self.spin_max_dets.value()
        self.parameters.use_custom_model = self.check_custom_model.isChecked()
        self.parameters.config_file = self.browse_custom_cfg.path
        self.parameters.model_weight_file = self.browse_custom_weights.path
        self.parameters.update = True
        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferYoloWorldWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_yolo_world"

    def create(self, param):
        # Create widget object
        return InferYoloWorldWidget(param, None)
