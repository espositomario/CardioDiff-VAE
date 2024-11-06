# Define the values of CT and R
CT_LIST=("ESC" "MES" "CP" "CM")
R_VALUES=(1 2)

# Iterate over CT and R values
for CT in "${CT_LIST[@]}"; do
    for R in "${R_VALUES[@]}"; do
    # Construct the path string
    path="/Volumes/MariusSSD/prj/crg/02_map_files/mm10/Wamstad/RNA_${CT}_${R}_Wamstad_2013_PE/accepted_hits.bam"
    # Construct the symlink name
    symlink="RNA_${CT}_${R}_Wamstad_2013_PE.link"
    # Create the symbolic link
    ln -s "$path" "$symlink"
    done
done