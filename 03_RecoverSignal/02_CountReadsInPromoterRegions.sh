$MOUSE10="mm10MouseGenomeDir/..."
# Problematic regions (Aspecific High Signal)
-> mm10_blacklist_v2_boyle.bed

wget https://github.com/Boyle-Lab/Blacklist/raw/master/lists/mm10-blacklist.v2.bed.gz
cat mm10-blacklist.v2.bed | tr ' ' '_' > $MOUSE10/mm10_blacklist_v2_boyle.bed



#----------exclude blacklisted regions from refGene_promoters_${DISTANCE}.txt----------
DISTANCE=2500
matchpeaks -v $MOUSE10/mm10_blacklist_v2_boyle.bed refGene_promoters_${DISTANCE}.bed blacklist_v2 refGene_promoters_${DISTANCE}
tail +2 blacklist_v2_refGene_promoters_${DISTANCE}_matchpeaks/only_refGene_promoters_${DISTANCE}.bed | cut -f1,2,3,5 | sort -k 4 > refGene_promoters_${DISTANCE}_clean.bed


# n= 31540 refGene_promoters_2500.bed
# n= 29480 refGene_promoters_2500_clean.bed



#----------------------recoverChIPlevels_parallel()----------------------
MAPFILES='BAMFilesDir/...'
FILE_NAME="../01_Mapping/data/FILE_NAMES_NARROW.list"
recoverChIPlevels_parallel() {
    local NAME=$1
    mkdir -p "recoverChIPlevels_promoters_2500"
    recoverChIPlevels -vn -l 250 -x "recoverChIPlevels_promoters_2500" "${MOUSE10}/ChromInfo.txt" "${MAPFILES}${NAME}.bam" "refGene_promoters_2500_clean.bed" "$NAME"
}
export -f recoverChIPlevels_parallel

parallel --progress -j 5 --load 80% recoverChIPlevels_parallel :::: $FILE_NAME
