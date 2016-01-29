#!/usr/bin/env bash

I3_BUILD=${HOME}/work/pingusoft_build_lion
I3_SRC=${HOME}/work/pingusoft/src
NUTAU=${HOME}/work/nu_tau_appearance/pisa
SRC_SCRIPT_DIR=${HOME}/work/pisa/pisa/analysis/llr
PBS_FILE=${NUTAU}/PBS/pisa_simp_15sys_0.045yr_LLR_8_by_16_MH.pbs.DC.in

PROJECT=`basename ${PBS_FILE} _simp_15sys_0.045yr_LLR_8_by_16_MH.pbs.DC.in`
JOB_NAME=$PROJECT
echo $PROJECT
INPUT_DIR=${HOME}/scratch/${PROJECT}/DC_simp_15sys_0.045yr_LLR_8_by_16_MH
OUTPUT_DIR=${HOME}/scratch/${PROJECT}/DC_simp_15sys_0.045yr_LLR_8_by_16_MH/output
LOG_DIR=${HOME}/scratch/${PROJECT}/DC_simp_15sys_0.045yr_LLR_8_by_16_MH/log
PBS_SCRIPT_DIR=${HOME}/work/PBS_script_pisa/DC_simp_15sys_0.045yr_LLR_8_by_16_MH
echo $PBS_SCRIPT_DIR
NFITS=1000
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

##${I3_BUILD}/env-shell.sh bash <<EOF
##for run in {100,101,102,103};
##for run in {0..`expr \${NFITS} / 10`};
bash <<EOF
for run in {1000..3500};
do
	##echo \$run
	BASENAME=LLR_005_MH
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

