echo "*******************************************"
echo "Running Training for aunet_gcl_rmi_occ"
echo "*******************************************"
bash scripts/thermalFaceDB/aunet/run_aunet_gcl_rmi_train_occ.sh train aunet_gcl_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

echo "*******************************************"
echo "Running Training for aunet_rmi_occ"
echo "*******************************************"
bash scripts/thermalFaceDB/aunet/run_aunet_rmi_train_occ.sh train aunet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

echo "*******************************************"
echo "Running Training for aunet_gcl_rmi"
echo "*******************************************"
bash scripts/thermalFaceDB/aunet/run_aunet_gcl_rmi_train.sh train aunet_gcl_rmi ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

echo "*******************************************"
echo "Running Validation for aunet_rmi_occ"
echo "*******************************************"
bash scripts/thermalFaceDB/aunet/run_aunet_rmi_train_occ.sh train aunet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

echo "*******************************************"
echo "Running Validation for aunet_gcl_rmi"
echo "*******************************************"
bash scripts/thermalFaceDB/aunet/run_aunet_gcl_rmi_train.sh train aunet_gcl_rmi ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

echo "*******************************************"
echo "Running Validation for aunet_gcl_rmi_occ"
echo "*******************************************"
bash scripts/thermalFaceDB/aunet/run_aunet_gcl_rmi_train_occ.sh train aunet_gcl_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Training for aunet_rmi"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_rmi_train.sh train aunet_rmi ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320

# echo "*******************************************"
# echo "Running Validation for aunet_rmi"
# echo "*******************************************"
# bash scripts/thermalFaceDB/aunet/run_aunet_rmi_train.sh train aunet_rmi ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320



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
