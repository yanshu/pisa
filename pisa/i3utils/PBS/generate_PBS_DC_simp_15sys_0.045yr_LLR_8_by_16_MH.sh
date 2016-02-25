#!/usr/bin/env bash

TAG=DC_simp_15sys_0.045yr_LLR_8_by_16

NUTAU=${HOME}/pisa/pisa
#SRC_SCRIPT_DIR=${HOME}/pisa/pisa/analysis/llr
SRC_SCRIPT_DIR=${HOME}/pisa
PBS_FILE=${NUTAU}/i3utils/PBS/pisa_simp_15sys_0.045yr_LLR_8_by_16_MH.pbs.DC.in

PROJECT=`basename ${PBS_FILE} _simp_15sys_0.045yr_LLR_8_by_16_MH.pbs.DC.in`
JOB_NAME=$PROJECT
echo $PROJECT
INPUT_DIR=${HOME}/scratch/${PROJECT}/${TAG}
OUTPUT_DIR=${HOME}/scratch/${PROJECT}/${TAG}/output
LOG_DIR=${HOME}/scratch/${PROJECT}/${TAG}/log
PBS_SCRIPT_DIR=${HOME}/work/PBS_script_pisa/${TAG}
echo $PBS_SCRIPT_DIR
mkdir -p ${OUTPUT_DIR} ${LOG_DIR} ${PBS_SCRIPT_DIR} || exit 1

echo "Input  directory ${INPUT_DIR} set"
echo "Output directory ${OUTPUT_DIR} created"
echo "Log    directory ${LOG_DIR} created"
echo "Script directory ${PBS_SCRIPT_DIR} created"

export PROJECT
export INPUT_DIR
export OUTPUT_DIR
export LOG_DIR
export PBS_FILE
export SRC_SCRIPT_DIR
export PBS_SCRIPT_DIR

bash <<EOF
for run in {1000..3000};
do
	##echo \$run
	BASENAME=LLR_8b16
	sed	-e "s|@@INPUT_FILE@@|\${in_file}|g" \
       	 	-e "s|@@OUTPUT_PROJECT@@|${OUTPUT_DIR}/\${PROJECT}|g" \
		-e "s|@@LOG_FILE@@|${LOG_DIR}/\${PROJECT}|g" \
       		-e "s|@@JOB_NAME@@|\${BASENAME}|g" \
       		-e "s|@@RUN@@|\${run}|g" \
       		-e "s|@@PATH@@|\${PATH}|g" \
       		-e "s|@@LD_LIBRARY_PATH@@|\${LD_LIBRARY_PATH}|g" \
       		-e "s|@@ICETRAY_CLASSPATH@@|\${ICETRAY_CLASSPATH}|g" \
       		-e "s|@@SRC_SCRIPT_DIR@@|\${SRC_SCRIPT_DIR}|g" \
       		-e "s|@@OUTPUT_DIR@@|\${OUTPUT_DIR}|g" \
		   ${PBS_FILE} \
		   > ${PBS_SCRIPT_DIR}/\${BASENAME}_\${run}.sh
done

EOF

echo " Done!"

cat <<EOF
===============================================================================

Please submit jobs to your preferred cluster!

Suggested command line:
\$ ~/scratch/qsub_wrapper.sh ${PBS_SCRIPT_DIR}/*.sh

===============================================================================
EOF

