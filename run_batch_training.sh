echo "*******************************************"
echo "Running Training for ResNet101_Dilation8_RMI"
echo "*******************************************"
bash scripts/thermalFaceDB/deeplab/run_r101_8_deeplabv3_train.sh train r101_8_base ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

echo "*******************************************"
echo "Running Training for ResNet101_Dilation8_RMI_Occ"
echo "*******************************************"
bash scripts/thermalFaceDB/deeplab/run_r101_8_deeplabv3_train_occ.sh train r101_8_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

echo "*******************************************"
echo "Running Training for ResNet101_Dilation8_RMI_SAMCL_Occ"
echo "*******************************************"
bash scripts/thermalFaceDB/deeplab/run_r101_8_deeplabv3_train_samcl_occ.sh train r101_8_samcl_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for ocnet_rmi"
# echo "*******************************************"
# bash scripts/thermalFaceDB/deeplab/run_r_101_ocnet_train_rmi.sh train ocnet_rmi ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for ocnet_rmi_occ"
# echo "*******************************************"
# bash scripts/thermalFaceDB/deeplab/run_r_101_ocnet_train_rmi_occ.sh train ocnet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for ocnet_samcl_rmi_occ"
# echo "*******************************************"
# bash scripts/thermalFaceDB/deeplab/run_r_101_ocnet_train_samcl_rmi_occ.sh train ocnet_samcl_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# bash scripts/thermalFaceDB/deeplab/run_r_101_ocnet_train_rmi_occ.sh val ocnet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# echo "*******************************************"
# echo "Running Training for unet_contrast_no_occ_oldAnn"
# echo "*******************************************"
# bash scripts/thermalFaceDB/unet/run_unet_contrast_train.sh train unet_contrast_no_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for unet_contrast_occ_oldAnn"
# echo "*******************************************"
# bash scripts/thermalFaceDB/unet/run_unet_contrast_train_occ.sh train unet_contrast_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for unet_contrast_mem_no_occ_oldAnn"
# echo "*******************************************"
# bash scripts/thermalFaceDB/unet/run_unet_contrast_mem_train.sh train unet_contrast_mem_no_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for unet_contrast_mem_occ_oldAnn"
# echo "*******************************************"
# bash scripts/thermalFaceDB/unet/run_unet_contrast_mem_train_occ.sh train unet_contrast_mem_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Validation"
# bash scripts/thermalFaceDB/unet/run_unet_contrast_train.sh val unet_contrast_no_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/unet/run_unet_contrast_train_occ.sh val unet_contrast_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/unet/run_unet_contrast_mem_train.sh val unet_contrast_mem_no_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/unet/run_unet_contrast_mem_train_occ.sh val unet_contrast_mem_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"

# echo "*******************************************"
# echo "Running Training for aunet_contrast_no_occ_oldAnn"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_contrast_train.sh train aunet_contrast_no_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for aunet_contrast_occ_oldAnn"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_contrast_train_occ.sh train aunet_contrast_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for aunet_contrast_mem_no_occ_oldAnn"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_contrast_mem_train.sh train aunet_contrast_mem_no_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for aunet_contrast_mem_occ_oldAnn"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_contrast_mem_train_occ.sh train aunet_contrast_mem_occ_oldAnn ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320


# # echo "*******************************************"
# # echo "Running Training for aunet_rmi"
# # echo "*******************************************"
# # bash scripts/thermalFaceDB/aunet/run_aunet_rmi_train.sh train aunet_rmi ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for aunet_gcl_rmi_occ"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_gcl_rmi_train_occ.sh train aunet_gcl_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for aunet_rmi_occ"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_rmi_train_occ.sh train aunet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# # echo "*******************************************"
# # echo "Running Training for aunet_gcl_rmi"
# # echo "*******************************************"
# # bash scripts/thermalFaceDB/aunet/run_aunet_gcl_rmi_train.sh train aunet_gcl_rmi ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# # echo "*******************************************"
# # echo "Running Validation for aunet_rmi"
# # echo "*******************************************"
# # bash scripts/thermalFaceDB/aunet/run_aunet_rmi_train.sh val aunet_rmi ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Validation for aunet_rmi_occ"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_rmi_train_occ.sh val aunet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# # echo "*******************************************"
# # echo "Running Validation for aunet_gcl_rmi"
# # echo "*******************************************"
# # bash scripts/thermalFaceDB/aunet/run_aunet_gcl_rmi_train.sh val aunet_gcl_rmi ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Validation for aunet_gcl_rmi_occ"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_gcl_rmi_train_occ.sh val aunet_gcl_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320




# bash scripts/thermalFaceDB/hrnet/run_h_48_d_8_hrnet_rmi_train.sh train hrnet_rmi_final ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/hrnet/run_h_48_d_8_hrnet_rmi_train_occ.sh train hrnet_rmi_occ_final ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/hrnet/run_h_48_d_8_hrnet_gcl_rmi_train.sh train hrnet_gcl_rmi_final ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/hrnet/run_h_48_d_8_hrnet_gcl_rmi_train_occ.sh train hrnet_gcl_rmi_occ_final ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/hrnet/run_h_48_d_8_hrnet_rmi_train.sh val hrnet_rmi_final ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/hrnet/run_h_48_d_8_hrnet_rmi_train_occ.sh val hrnet_rmi_occ_final ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/hrnet/run_h_48_d_8_hrnet_gcl_rmi_train.sh val hrnet_gcl_rmi_final ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/hrnet/run_h_48_d_8_hrnet_gcl_rmi_train_occ.sh val hrnet_gcl_rmi_occ_final ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320


# bash scripts/thermalFaceDB/unet/run_unet_rmi_train_occ.sh train unet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/unet/run_unet_rmi_gcl_train.sh train unet_rmi_gcl ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/unet/run_unet_rmi_gcl_train_occ.sh train unet_rmi_gcl_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/unet/run_unet_rmi_train_occ.sh val unet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/unet/run_unet_rmi_gcl_train.sh val unet_rmi_gcl ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
# bash scripts/thermalFaceDB/unet/run_unet_rmi_gcl_train_occ.sh val unet_rmi_gcl_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

