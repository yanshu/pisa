#!/bin/bash

BASEDIR=$(dirname "$0")
PISA=$BASEDIR/..
echo "PISA=$PISA"


echo "=============================================================================="
echo "Running test_consistency_with_pisa2.py"
echo "=============================================================================="
$BASEDIR/test_consistency_with_pisa2.py -v
echo "------------------------------------------------------------------------------"
echo "Finished Running test_consistency_with_pisa2.py"
echo "------------------------------------------------------------------------------"
echo ""
echo ""


echo "=============================================================================="
echo "Running test_example_pipelines.py"
echo "=============================================================================="
$BASEDIR/test_example_pipelines.py -v
echo "------------------------------------------------------------------------------"
echo "Finished Running test_example_pipelines.py"
echo "------------------------------------------------------------------------------"
echo ""
echo ""


# TODO: all files except setup.py and __init__.py that are listed below should
# have a command-line test defined further down in this script (i.e., these are
# scripts that require specific command-line arguments)
for f in `find $PISA/pisa -name "*.py"`
do
	BN=$(basename "$f")
	if [[ "$BN" == test_* ]];then continue;fi
	if [[ "$f" == *pisa/scripts/* ]];then continue;fi
	if [ "$BN" == "__init__.py" ];then continue;fi
	if [ "$BN" == "setup.py" ];then continue;fi
	if [ "$BN" == hypo_testing.py ];then continue;fi
	if [ "$BN" == hypo_testing_postprocess.py ];then continue;fi
	if [ "$BN" == profile_scan.py ];then continue;fi
	if [ "$BN" == theta23_octant_postprocess.py ];then continue;fi
	if [ "$BN" == profile_llh_analysis.py ];then continue;fi
	if [ "$BN" == nutau_analysis.py ];then continue;fi
	if [ "$BN" == theta23_octant.py ];then continue;fi
	if [ "$BN" == theta23_NMO_2.py ];then continue;fi
	if [ "$BN" == pipeline.py ];then continue;fi
	if [ "$BN" == distribution_maker.py ];then continue;fi
	if [ "$BN" == genie.py ];then continue;fi
	if [ "$BN" == roounfold.py ];then continue;fi
	if [ "$BN" == crossSections.py ];then continue;fi

	echo "=============================================================================="
	echo "Running python $BN at abs path"
    echo "  `realpath $f`"
	echo "=============================================================================="
	python $f || FAILURE=true
	echo "------------------------------------------------------------------------------"
	echo "Finished running python $BN"
	echo "------------------------------------------------------------------------------"
	echo ""
	echo ""
	sleep 1
done


OUTDIR=/tmp/compare_cpu_gpu
echo "=============================================================================="
echo "Running compare.py, CPU vs. GPU pipeline settings, FP64, no plotting."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
PISA_FTYPE=float64 $PISA/pisa/scripts/compare.py \
	--ref settings/pipeline/example.cfg \
	--ref-label 'cpu' \
	--test settings/pipeline/example_gpu.cfg \
	--test-label 'gpu' \
	--outdir $OUTDIR \
	-v


OUTDIR=/tmp/compare_cpu_gpu
echo "=============================================================================="
echo "Running compare.py, CPU vs. GPU pipeline settings, FP64."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
PISA_FTYPE=float64 $PISA/pisa/scripts/compare.py \
	--ref settings/pipeline/example.cfg \
	--ref-label 'cpu' \
	--test settings/pipeline/example_gpu.cfg \
	--test-label 'gpu' \
	--outdir $OUTDIR \
	--pdf --png -v


OUTDIR_IH=/tmp/ih_pipeline
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with ih selected."
echo "Storing results to"
echo "  $OUTDIR_IH"
echo "=============================================================================="
PISA_FTYPE=float64 $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example.cfg \
	--select "ih" \
	-d $OUTDIR_IH \
	--pdf --png -v

OUTDIR_NH=/tmp/nh_pipeline
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with nh selected."
echo "Storing results to"
echo "  $OUTDIR_NH"
echo "=============================================================================="
PISA_FTYPE=float64 $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example.cfg \
	--select "nh" \
	-d $OUTDIR_NH \
	--pdf --png -v

OUTDIR_NH_DIST_MAKER=/tmp/nh_dist_maker
echo "=============================================================================="
echo "Running distribution_maker.py with example.cfg, with nh selected."
echo "Storing results to"
echo "  $OUTDIR_NH_DIST_MAKER"
echo "=============================================================================="
PISA_FTYPE=float64 $PISA/pisa/core/distribution_maker.py \
	-p settings/pipeline/example.cfg \
	--select "nh" \
	-d $OUTDIR_NH_DIST_MAKER \
	--pdf --png -v

OUTDIR=/tmp/compare_nh_to_ih
echo "=============================================================================="
echo "Running compare.py, nh vs. ih MapSets produced above WITH plots."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
$PISA/pisa/scripts/compare.py \
	--ref $OUTDIR_IH/*.json.bz2 \
	--ref-label 'ih' \
	--test $OUTDIR_NH/*.json.bz2 \
	--test-label 'nh' \
	--outdir $OUTDIR \
	--pdf --png -v


OUTDIR_NH_FP32_CPU=/tmp/nh_pipeline_fp32_cpu
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with nh selected, FP32/CPU."
echo "Storing results to"
echo "  $OUTDIR_NH_FP32_CPU"
echo "=============================================================================="
PISA_FTYPE=float32 $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example.cfg \
	--select "nh" \
	-d $OUTDIR_NH_FP32_CPU \
	-v

OUTDIR_NH_FP32_GPU=/tmp/nh_pipeline_fp32_gpu
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with nh selected, FP32/GPU."
echo "Storing results to"
echo "  $OUTDIR_NH_FP32_GPU"
echo "=============================================================================="
PISA_FTYPE=float32 $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example_gpu.cfg \
	--select "nh" \
	-d $OUTDIR_NH_FP32_GPU \
	-v

OUTDIR_NH_FP64_GPU=/tmp/nh_pipeline_fp64_gpu
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with nh selected, FP64/GPU."
echo "Storing results to"
echo "  $OUTDIR_NH_FP64_GPU"
echo "=============================================================================="
PISA_FTYPE=float64 $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example_gpu.cfg \
	--select "nh" \
	-d $OUTDIR_NH_FP64_GPU \
	-v

OUTDIR=/tmp/compare_fp32_cpu_to_fp64_cpu
echo "=============================================================================="
echo "Running compare.py, fp32/cpu vs. fp64/cpu MapSets produced above."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
$PISA/pisa/scripts/compare.py \
	--ref $OUTDIR_NH/*.json.bz2 \
	--ref-label 'fp64_cpu' \
	--test $OUTDIR_NH_FP32_CPU/*.json.bz2 \
	--test-label 'fp32_cpu' \
	--outdir $OUTDIR \
	--pdf -v

OUTDIR=/tmp/compare_fp32_gpu_to_fp64_cpu
echo "=============================================================================="
echo "Running compare.py, fp32/gpu vs. fp64/cpu MapSets produced above."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
$PISA/pisa/scripts/compare.py \
	--ref $OUTDIR_NH/*.json.bz2 \
	--ref-label 'fp64_cpu' \
	--test $OUTDIR_NH_FP32_GPU/*.json.bz2 \
	--test-label 'fp32_gpu' \
	--outdir $OUTDIR \
	--pdf -v

OUTDIR=/tmp/compare_fp64_gpu_to_fp64_cpu
echo "=============================================================================="
echo "Running compare.py, fp64/gpu vs. fp64/cpu MapSets produced above."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
$PISA/pisa/scripts/compare.py \
	--ref $OUTDIR_NH/*.json.bz2 \
	--ref-label 'fp64_cpu' \
	--test $OUTDIR_NH_FP64_GPU/*.json.bz2 \
	--test-label 'fp64_gpu' \
	--outdir $OUTDIR \
	--pdf -v

OUTDIR=/tmp/compare_fp32_gpu_to_fp64_gpu
echo "=============================================================================="
echo "Running compare.py, fp32/gpu vs. fp64/gpu MapSets produced above."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
$PISA/pisa/scripts/compare.py \
	--ref $OUTDIR_NH/*.json.bz2 \
	--ref-label 'fp64_gpu' \
	--test $OUTDIR_NH_FP32_GPU/*.json.bz2 \
	--test-label 'fp32_gpu' \
	--outdir $OUTDIR \
	--pdf -v


OUTDIR=/tmp/hypo_testing_test
echo "=============================================================================="
echo "Running hypo_testing.py, basic NMO Asimov analysis (not necessarily accurate)"
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
PISA_FTYPE=float64 $PISA/pisa/analysis/hypo_testing.py \
	--h0-pipeline settings/pipeline/example.cfg \
	--h0-param-selections="ih" \
	--h1-param-selections="nh" \
	--data-param-selections="nh" \
	--data-is-mc \
	--minimizer-settings settings/minimizer/bfgs_settings_fac1e11_eps1e-4_mi20.json \
	--metric=chi2 \
	--logdir $OUTDIR \
	--pprint -v --allow-dirty
