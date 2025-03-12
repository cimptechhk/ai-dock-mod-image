#!/bin/bash

# This file will be sourced in init.sh

# https://raw.githubusercontent.com/ai-dock/comfyui/main/config/provisioning/default.sh

# Packages are installed after nodes so we can fix them...

#DEFAULT_WORKFLOW="https://..."

APT_PACKAGES=(
    #"package-1"
    #"package-2"
)

PIP_PACKAGES=(
    "segment-anything"
    "scikit-image"
    "scikit-learn"
    "piexif"
    "opencv-python>=4.7.0.72"
    "scipy"
    "numpy<2"
    "dill"
    "matplotlib"
    "onnxruntime"
    "timm>=1.0.7"
    "addict"
    "yapf"
    "pillow==9.5.0"
    "diffusers"
    "accelerate>=0.30.0"
    "transformers>=4.43.2"
    "opencv-python-headless"
    "imageio"
    "imageio-ffmpeg>=0.5.1"
    "lmdb>1.4.1"
    "rich>=13.7.1"
    "albumentations>=1.4.16"
    "ultralytics"
    "tyro==0.8.5"
    "torch"
    "opencv-contrib-python"
    "pymatting"
    "colour-science"
    "blend_modes"
    "huggingface_hub>=0.23.4"
    "loguru"
    "filelock"
    "einops"
    "torchvision"
    "pyyaml"
    "python-dateutil"
    "mediapipe"
    "svglib"
    "fvcore"
    "omegaconf"
    "ftfy"
    "yacs"
    "trimesh[easy]"
    "fairscale"
    "pycocoevalcap"
    "qrcode[pil]"
    "pytorch_lightning"
    "kornia"
    "pydantic"
    "boto3>=1.34.101"
    "color-matcher"
    "mss"
    "insightface==0.7.3"
    "onnx>=1.14.0"
    "numexpr"
    "simpleeval"
    "ninja"
    "/tmp/pip/nvdiffrast/"
)

NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    "https://github.com/chflame163/ComfyUI_LayerStyle"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/1038lab/ComfyUI-RMBG"
    "https://github.com/storyicon/comfyui_segment_anything"
    "https://github.com/spacepxl/ComfyUI-Image-Filters"
    "https://github.com/chflame163/ComfyUI_LayerStyle_Advance"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/chflame163/ComfyUI_CatVTON_Wrapper"
    "https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait"
    "https://github.com/chrisgoringe/cg-use-everywhere"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/sipherxyz/comfyui-art-venture"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    "https://github.com/Gourieff/comfyui-reactor-node"
    "https://github.com/jakechai/ComfyUI-JakeUpgrade"
    "https://github.com/ltdrdata/ComfyUI-Impact-Subpack"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/kijai/ComfyUI-LivePortraitKJ"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
    "https://github.com/giriss/comfy-image-saver"
    "https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes"
    "https://github.com/kijai/ComfyUI-CogVideoXWrapper"
    "https://github.com/kijai/ComfyUI-Florence2"
    "https://github.com/un-seen/comfyui-tensorops"
    "https://github.com/lldacing/ComfyUI_PuLID_Flux_ll"
    "https://github.com/chengzeyi/Comfy-WaveSpeed"
    "https://github.com/city96/ComfyUI-GGUF"
    "https://github.com/jags111/efficiency-nodes-comfyui"
    "https://github.com/ssitu/ComfyUI_UltimateSDUpscale"
    "https://github.com/SeargeDP/SeargeSDXL"
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes"
    "https://github.com/M1kep/Comfy_KepListStuff"
    "https://github.com/jjkramhoeft/ComfyUI-Jjk-Nodes"
    "https://github.com/SozeInc/ComfyUI_Soze"
    "https://github.com/MoonHugo/ComfyUI-BiRefNet-Hugo"
    "https://github.com/mirabarukaso/ComfyUI_Mira"
    "https://github.com/stormcenter/ComfyUI-AutoSplitGridImage"
)

WORKFLOWS=(
    "https://huggingface.co/cimp-tech/ComfyUI-workflows/resolve/main/Advanced%20Image%20Generation.json"
    "https://huggingface.co/cimp-tech/ComfyUI-workflows/resolve/main/Background%20Remover.json"
    "https://huggingface.co/cimp-tech/ComfyUI-workflows/resolve/main/Changing%20Cloth.json"
    "https://huggingface.co/cimp-tech/ComfyUI-workflows/resolve/main/Create%20Character%20%26%20Character%20sheet.json"
    "https://huggingface.co/cimp-tech/ComfyUI-workflows/resolve/main/Image%20Pass%20Extraction%20%26%20Change%20Style.json"
    "https://huggingface.co/cimp-tech/ComfyUI-workflows/resolve/main/Character_sheet_PuLID.json"
    "https://huggingface.co/cimp-tech/ComfyUI-workflows/resolve/main/Cog_Hunyuan_V2V.json"
)

declare -A HF_REPOS=(
    ["https://cimptech:${HF_TOKEN}@huggingface.co/cimp-tech/CatVTON"]="/${WORKSPACE}/ComfyUI/models/CatVTON"
    # layer_style START
    ["https://huggingface.co/briaai/RMBG-2.0"]="/${WORKSPACE}/ComfyUI/models/rmbg/RMBG-2.0"
    ["https://huggingface.co/PramaLLC/BEN"]="/${WORKSPACE}/ComfyUI/models/rmbg/BEN"
    # layer_style END
    ["https://cimptech:${HF_TOKEN}@huggingface.co/cimp-tech/insightface"]="/${WORKSPACE}/ComfyUI/models/insightface"
    ["https://cimptech:${HF_TOKEN}@huggingface.co/cimp-tech/ipadapter"]="/${WORKSPACE}/ComfyUI/models/ipadapter"
    #["https://cimptech:${HF_TOKEN}@huggingface.co/cimp-tech/test"]="/${WORKSPACE}/"
)

CHECKPOINT_MODELS=(
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    #"https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt"
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
    "https://civitai.com/api/download/models/318939"
    "https://civitai.com/api/download/models/429454"
    "https://huggingface.co/Fadedfragger/ElixirProject/resolve/main/elixirproject_v16.safetensors"
)

DIFFUSION_MODELS=(

)

CLIP_MODELS=(
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
)

CLIP_VISION_MODELS=(
    "https://huggingface.co/shiertier/clip_vision/resolve/main/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"
    "https://huggingface.co/cimp-tech/IPAdapter_plus_clip_vision/resolve/main/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    "https://huggingface.co/cimp-tech/IPAdapter_plus_clip_vision/resolve/main/clip-vit-large-patch14-336.bin"
    "https://huggingface.co/cimp-tech/IPAdapter_plus_clip_vision/resolve/main/clip-vit-large-patch14.bin"
)

STYLE_MODELS=(
    "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors"
)

UNET_MODELS=(
    "https://huggingface.co/lllyasviel/FLUX.1-dev-gguf/resolve/d4374ef1edc2e1c0ef8907e57eb2588834170c96/flux1-dev-Q4_K_S.gguf"
)

LORA_MODELS=(
    "https://civitai.com/api/download/models/10224"
)

VAE_MODELS=(
    "https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
)

ESRGAN_MODELS=(
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
    "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
    "https://huggingface.co/Akumetsu971/SD_Anime_Futuristic_Armor/resolve/main/4x_NMKD-Siax_200k.pth"
)

CONTROLNET_MODELS=(
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_mid.safetensors"
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_depth_mid.safetensors?download"
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_openpose.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_depth-fp16.safetensors"
    "https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_depth_fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_hed-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_mlsd-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_normal-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_scribble-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_seg-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_color-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_depth-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_keypose-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_seg-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_sketch-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_style-fp16.safetensors"
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors"
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors" 
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors"
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors"
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors"
)

IPADAPTER_LORA_MODELS=(
    "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors"
    "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
)

VITMATTE_MODELS=(
    "https://huggingface.co/shiertier/vitmatte/resolve/main/.gitattributes"
    "https://huggingface.co/shiertier/vitmatte/resolve/main/README.md"
    "https://huggingface.co/shiertier/vitmatte/resolve/main/config.json"
    "https://huggingface.co/shiertier/vitmatte/resolve/main/model.safetensors"
    "https://huggingface.co/shiertier/vitmatte/resolve/main/preprocessor_config.json"
    "https://huggingface.co/shiertier/vitmatte/resolve/main/pytorch_model.bin"
)

RMBG_1_4_MODELS=(
    "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth"
)

INSPYRENET_MODELS=(
    "https://huggingface.co/1038lab/inspyrenet/resolve/main/inspyrenet.safetensors"
)

ULTRALYTICS_BBOX_MODELS=(
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov9c.pt"

)

ULTRALYTICS_SEGM_MODELS=(
    "https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt"
    "https://huggingface.co/muciz/iseng/resolve/ac3f4bba2423d5e29af5cbd73eb2fc0e433e0c2f/PitEyeDetailer-v2-seg.pt"
)

FLUX1_DIFFUSION_MODELS=(
    "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors"
)

FLUX1_VAE_MODELS=(
    "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"
)

FLUX_TEXT_ENCODERS=(
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
)

FLUX1_CONTROLNET_MODELS=(
    "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux_shakker_labs_union_pro-fp8_e4m3fn.safetensors"
)

PULID_MODELS=(
    "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors"
)

UPSCALE_MODELS=(
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth"
    "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    if [[ ! -d /opt/environments/python ]]; then 
        export MAMBA_BASE=true
    fi
    source /opt/ai-dock/etc/environment.sh
    source /opt/ai-dock/bin/venv-set.sh comfyui

    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_nodes
    provisioning_get_pip_packages
    provisioning_get_HF_repo
    # GET WORKFLOWS START
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/user/default/workflows" \
        "${WORKFLOWS[@]}"
    # GET WORKFLOWS END
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/checkpoints" \
        "${CHECKPOINT_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/diffusion_models" \
        "${DIFFUSION_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/diffusion_models/FLUX1" \
        "${FLUX1_DIFFUSION_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/style_models" \
        "${STYLE_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/text_encoders/t5" \
        "${FLUX_TEXT_ENCODERS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/unet" \
        "${UNET_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/loras" \
        "${LORA_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/controlnet" \
        "${CONTROLNET_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/controlnet/FLUX.1" \
        "${FLUX1_CONTROLNET_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/vae" \
        "${VAE_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/vae/FLUX1" \
        "${FLUX1_VAE_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/clip" \
        "${CLIP_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/clip_vision" \
        "${CLIP_VISION_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/esrgan" \
        "${ESRGAN_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/upscale_models" \
        "${UPSCALE_MODELS[@]}"
    # PuLID
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/pulid" \
        "${PULID_MODELS[@]}" 
    # vitmatte
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/vitmatte" \
        "${VITMATTE_MODELS[@]}"    
    # ipadapter
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/loras/ipadapter" \
        "${IPADAPTER_LORA_MODELS[@]}"
    # ultralytics
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/ultralytics/bbox" \
        "${ULTRALYTICS_BBOX_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/ultralytics/segm" \
        "${ULTRALYTICS_SEGM_MODELS[@]}"
    # layer_style START
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/rmbg/RMBG-1.4" \
        "${RMBG_1_4_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/rmbg/INSPYRENET" \
        "${INSPYRENET_MODELS[@]}"
    # layer_style END
    provisioning_print_end
}

function pip_install() {
    if [[ -z $MAMBA_BASE ]]; then
            sudo "$COMFYUI_VENV_PIP" install --no-cache-dir "$@"
        else
            sudo micromamba run -n comfyui pip install --no-cache-dir "$@"
        fi
}

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
            sudo $APT_INSTALL ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
            pip_install ${PIP_PACKAGES[@]}
    fi
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="/${WORKSPACE}/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                   pip_install -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                pip_install -r "${requirements}"
            fi
        fi
    done
}

function provisioning_get_HF_repo() {
    for repo in "${!HF_REPOS[@]}"; do
        path="${HF_REPOS[$repo]}"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
        fi
    done
}

function provisioning_get_default_workflow() {
    if [[ -n $DEFAULT_WORKFLOW ]]; then
        workflow_json=$(curl -s "$DEFAULT_WORKFLOW")
        if [[ -n $workflow_json ]]; then
            echo "export const defaultGraph = $workflow_json;" > /opt/ComfyUI/web/scripts/defaultGraph.js
        fi
    fi
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    
    dir="$1"
    mkdir -p "$dir"
    shift
    arr=("$@")
    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
    if [[ $DISK_GB_ALLOCATED -lt $DISK_GB_REQUIRED ]]; then
        printf "WARNING: Your allocated disk size (%sGB) is below the recommended %sGB - Some models will not be downloaded\n" "$DISK_GB_ALLOCATED" "$DISK_GB_REQUIRED"
    fi
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Web UI will start now\n\n"
}

function provisioning_has_valid_hf_token() {
    [[ -n "$HF_TOKEN" ]] || return 1
    url="https://huggingface.co/api/whoami-v2"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_has_valid_civitai_token() {
    [[ -n "$CIVITAI_TOKEN" ]] || return 1
    url="https://civitai.com/api/v1/models?hidden=1&limit=1"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

# Download from $1 URL to $2 file path
function provisioning_download() {
    if [[ -n $HF_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif 
        [[ -n $CIVITAI_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [[ -n $auth_token ]];then
        wget --header="Authorization: Bearer $auth_token" -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    fi
}

provisioning_start
