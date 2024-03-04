import requests
import os


model_info = {
    "yolo_world_s": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-18bea4d2.pth",
        "config": "yolo_world_s_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    },
    "yolo_world_m": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_m_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-2b7bd1be.pth",
        "config": "yolo_world_m_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    },
    "yolo_world_l": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth",
        "config": "yolo_world_m_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    },
    "yolo_world_l_plus": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-7a5eea3b.pth",
        "config": "yolo_world_l_dual_vlpan_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    },
    "yolo_world_x": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_x_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-8cf6b025.pth",
        "config": "yolo_world_x_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    },
    "yolo_world_v2_s": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth",
        "config": "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    },
    "yolo_world_v2_m": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth",
        "config": "yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    },
    "yolo_world_v2_l": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth",
        "config": "yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    },
    "yolo_world_v2_l_plus": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth",
        "config": "yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    },
    "yolo_world_v2_x": {
        "url": "https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.pth",
        "config": "yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    }
}

def download_model_weights(model_name):
    model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder, exist_ok=True)


    if model_name in model_info:
        url = model_info[model_name]["url"]
        config = model_info[model_name]["config"]
        
        # Adjust URL if it's a direct link to a file in Hugging Face (replace "/blob/" with "/resolve/")
        if "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        
        # Extract filename from URL
        filename = url.split("/")[-1]

        save_path = os.path.join(model_folder, filename)
        
        # Download the model weights
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {filename} to {save_path}")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
        
        return save_path, config
    else:
        print(f"Model name '{model_name}' not recognized.")
        return None


if __name__ == "__main__":
    model_name = "yolo_world_s"
    config_name = download_model_weights(model_name)
    if config_name:
        print(f"Model Config Name: {config_name}")