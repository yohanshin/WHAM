#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL Neutral model
echo -e "\nYou need to register at https://smplify.is.tue.mpg.de"
read -p "Username (SMPLify):" username
read -p "Password (SMPLify):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p dataset/body_models/smpl
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O './dataset/body_models/smplify.zip' --no-check-certificate --continue
unzip dataset/body_models/smplify.zip -d dataset/body_models/smplify
mv dataset/body_models/smplify/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_NEUTRAL.pkl
rm -rf dataset/body_models/smplify
rm -rf dataset/body_models/smplify.zip

# SMPL Male and Female model
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O './dataset/body_models/smpl.zip' --no-check-certificate --continue
unzip dataset/body_models/smpl.zip -d dataset/body_models/smpl
mv dataset/body_models/smpl/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_FEMALE.pkl
mv dataset/body_models/smpl/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_MALE.pkl
rm -rf dataset/body_models/smpl/smpl
rm -rf dataset/body_models/smpl.zip

# Auxiliary SMPL-related data
wget "https://drive.google.com/uc?id=1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ&export=download&confirm=t" -O 'dataset/body_models.tar.gz'
tar -xvf dataset/body_models.tar.gz -C dataset/
rm -rf dataset/body_models.tar.gz

# Checkpoints
mkdir checkpoints
gdown "https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t" -O 'checkpoints/wham_vit_w_3dpw.pth.tar'
gdown "https://drive.google.com/uc?id=19qkI-a6xuwob9_RFNSPWf1yWErwVVlks&export=download&confirm=t" -O 'checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'
gdown "https://drive.google.com/uc?id=1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y&export=download&confirm=t" -O 'checkpoints/hmr2a.ckpt'
gdown "https://drive.google.com/uc?id=1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo&export=download&confirm=t" -O 'checkpoints/yolov8x.pt'
gdown "https://drive.google.com/uc?id=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso&export=download&confirm=t" -O 'checkpoints/vitpose-h-multi-coco.pth'

# Demo videos
gdown "https://drive.google.com/uc?id=1KjfODCcOUm_xIMLLR54IcjJtf816Dkc7&export=download&confirm=t" -O 'examples.tar.gz'
tar -xvf examples.tar.gz
rm -rf examples.tar.gz

