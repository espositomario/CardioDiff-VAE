
k=80
DIR="${HOME}/crg/07_CardioDiff/04_Models/data/6D_ALL_X_FC_z_analysis/GMM_VAE_${k}"
BAM_IDs="${DIR}/TSSplots/BAM_IDs_Rep1.list"
MAPFILES="/Volumes/MariusSSD/prj/crg/02_map_files/mm10/Wamstad"
#

produceTSSplots_parallel() {
    local BAM=$2
    local n=$1
    local DIR=$3
    local MAPFILES=$4
    produceTSSplots -w 50 -v -l 250 -x "${DIR}/TSSplots" "${MOUSE10}/ChromInfo.txt" "${MOUSE10}/refGene.txt" "${MAPFILES}/${BAM}.bam" "${DIR}/cluster_${n}.list" "C${n}_${BAM}" 2500
    #echo "produceTSSplots -v -l 250 -x "${DIR}/TSSplots" "${MOUSE10}/ChromInfo.txt" "${MOUSE10}/refGene.txt" "${MAPFILES}/${BAM}.bam" "${DIR}/cluster_${n}.list" "C${n}_${BAM}" 2500"
}
export -f produceTSSplots_parallel

parallel --progress -j 4 --load 80% produceTSSplots_parallel ::: $(seq 0 $((k-1))) :::: $BAM_IDs ::: ${DIR} ::: ${MAPFILES}
