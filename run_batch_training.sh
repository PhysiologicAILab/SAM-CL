bash scripts/thermalFaceDB/unet/run_unet_rmi_train_occ.sh train unet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
bash scripts/thermalFaceDB/unet/run_unet_rmi_gcl_train.sh train unet_rmi_gcl ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
bash scripts/thermalFaceDB/unet/run_unet_rmi_gcl_train_occ.sh train unet_rmi_gcl_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
bash scripts/thermalFaceDB/unet/run_unet_rmi_train_occ.sh val unet_rmi_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
bash scripts/thermalFaceDB/unet/run_unet_rmi_gcl_train.sh val unet_rmi_gcl ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320
bash scripts/thermalFaceDB/unet/run_unet_rmi_gcl_train_occ.sh val unet_rmi_gcl_occ ~/dev/data/ThermalFaceDBx320 ~/dev/data/ThermalFaceDBx320