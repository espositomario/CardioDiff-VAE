#!/usr/bin/perl -w                                #
#                                                 #
# Script ProcessRNAseq.pl                         #
#                                                 #
#                                                 #
# Input : GENOME, FASTQ file, NAME                # 
# Output: Mapping+UCSCprofile+RPKMs/FPKMs         #
#                                                 #
#                                                 #
# by Enrique Blanco @ CRG (2018)                  #
###################################################

use strict;
use Getopt::Std;
use Term::ANSIColor;

#DEFINEs
my $TRUE = 1;
my $FALSE = 0;
my $GENOME = "";
my $MAPS = "\$MAPFILES";
my $INFO = "info";


## Step 0. Reading arguments
my %opt;
my $start;
my ($genome,$version,$fastq_file1,$fastq_file2,$name);
my $n_files;
my $newdir;

(getopts('vsp',\%opt)) or print_error("parser: Problems reading options\n");

print_mess("ProcessRNAseq.pl by Enrique Blanco @ CRG (2018)\n\n");
print_mess("Stage 0.  Reading options");

$n_files = $#ARGV+1;
($n_files == 5) or print_error("USAGE: Five arguments are required but $n_files are provided!\nProcessRNAseq.pl -vsp <GENOME> <VERSION> <FASTQ1> <FASTQ2> <NAME>\n");
($genome,$version,$fastq_file1,$fastq_file2,$name) = @ARGV;

# 0.1 Save the starting time
$start = time();

# 0.2 Control of the genome
if ($genome eq "mouse")
{
    if ($version eq "mm9")
    {
        $GENOME = "\$MOUSE9";
        $MAPS = $MAPS."/mm9";
    }
    else
    {
        if ($version eq "mm10")
        {
           $GENOME = "\$MOUSE10";
           $MAPS = $MAPS."/mm10";
        }
        else
        {
            print_error("Please, use mm9 or mm10 as genome assembly for mouse ($version)");
        }
    }
}
else
{
    if ($genome eq "human")
    {
        if ($version eq "hg19")
        {
            $GENOME = "\$HUMAN19";
            $MAPS = $MAPS."/hg19";
        }
        else
        {
            if ($version eq "hg38")
            {
                $GENOME = "\$HUMAN38";
                $MAPS = $MAPS."/hg38";
            }
            else
            {
                print_error("Please, use hg19 or hg38 as genome assembly for human ($version)");
            }
        }
    }
    else
    {
        print_error("Please, select <human> or <mouse> for the genome mapping");
    }
}
print_ok();
##

# 0.3 Prepare the output folders (if necessary)
# create the info/ folder
$newdir = $INFO;
print_mess("Creating the $INFO directory\n");
mkdir($newdir) or print_mess("It is already existing\n");

## Step 1. Mapping the raw data in this genome
my $command;
my $NT = 6;  # Set the number of threads to 6

print_mess("Stage 1.  Mapping the RNAseq sample/s ($fastq_file1,$fastq_file2 -> $MAPS/$name/accepted_hits.bam)\n");

# single-end or pair-end mapping
if (exists($opt{p}))
{
    $command = "tophat2 --zpacker pigz --no-coverage-search --mate-inner-dist 176 --mate-std-dev 11 -p $NT -g 1 -G $GENOME/refGene.gtf -o $MAPS/$name --library-type=fr-firststrand $GENOME/genome $fastq_file1 $fastq_file2 2> info/tophat_$name.txt";
}
else
{
    $command = "tophat --no-coverage-search -p 4 -g 1 -G $GENOME/refGene.gtf -o $MAPS/$name $GENOME/genome $fastq_file1 2> info/tophat_$name.txt";
}

print_mess("Running $command");
system($command);
print_ok();
##

## Step 2. Obtain the statistics of the mapping by tophat
my $flagstat_sample;

print_mess("Stage 2.  Obtaining the statistics ($MAPS/$name/accepted_hits.bam)\n");

$flagstat_sample = "$INFO/flagstat_$name.txt";
$command = "samtools flagstat $MAPS/$name/accepted_hits.bam > $flagstat_sample";
print_mess("Running $command");
system($command);

$command = "cat $flagstat_sample";
print_mess("Running $command\n");
system($command);
print_ok();
##

## Step 3. Building the RNAseq profile/s for UCSC
my $profile;

print_mess("Stage 3.  Building the normalized profile/s for UCSC ($MAPS/$name/accepted_hits.bam -> $name)\n");
if (exists($opt{p}))
{
    $command = "samtools view -h -o $MAPS/$name/accepted_hits.sam $MAPS/$name/accepted_hits.bam";
    print_mess("Running $command");
    system($command);
    print_ok();

    $command = "grep \"XS:A:+\" $MAPS/$name/accepted_hits.sam > $MAPS/$name/accepted_hits_XSplus.sam";
    print_mess("Running $command");
    system($command);
    print_ok();

    $command = "grep \"XS:A:-\" $MAPS/$name/accepted_hits.sam > $MAPS/$name/accepted_hits_XSminus.sam";
    print_mess("Running $command");
    system($command);
    print_ok();

    $command = "samtools view -H $MAPS/$name/accepted_hits.bam > $MAPS/$name/header.sam";
    print_mess("Running $command");
    system($command);
    print_ok();

    $command = "cat $MAPS/$name/header.sam $MAPS/$name/accepted_hits_XSplus.sam > $MAPS/$name/accepted_hits_XSplusH.sam";
    print_mess("Running $command");
    system($command);
    print_ok();

    $command = "cat $MAPS/$name/header.sam $MAPS/$name/accepted_hits_XSminus.sam > $MAPS/$name/accepted_hits_XSminusH.sam";
    print_mess("Running $command");
    system($command);
    print_ok();

    $profile = $name."_plus";
    $command = "buildChIPprofile -p $GENOME/ChromInfo.txt $MAPS/$name/accepted_hits_XSplusH.sam $profile";
    print_mess("Running $command");
    system($command);
    print_ok();

    $profile = $name."_minus";
    $command = "buildChIPprofile -p $GENOME/ChromInfo.txt $MAPS/$name/accepted_hits_XSminusH.sam $profile";
    print_mess("Running $command");
    system($command);
    print_ok();

    $command = "rm -f $MAPS/$name/*.sam";
    print_mess("Running $command");
    system($command);
    print_ok();
    ##
}
else
{
    $command = "buildChIPprofile $GENOME/ChromInfo.txt $MAPS/$name/accepted_hits.bam $name";
    print_mess("Running $command");
    system($command);
    print_ok();
}


## Step 4. Calculate the RPKMs/FPKMs of the experiment
# my ($cufflinks1,$cufflinks2);
# 
# print_mess("Stage 4.  Calculating the RPKMs/FPKMs ($MAPS/$name/accepted_hits.bam -> RPKMs/FPKMs)\n");
# 
# $cufflinks1 = $name."_cufflinks";
# $cufflinks2 = $INFO."/cufflinks_".$name.".txt";
# 
# $command = "cufflinks --max-bundle-frags 5000000 -p 1 -G $GENOME/refGene.gtf -o $cufflinks1 $MAPS/$name/accepted_hits.bam 2> $cufflinks2";
# print_mess("Running $command");
# system($command);
# print_ok();


## Step 5. Finishing successful program execution
my $stop;

# Save the ending time
$stop = time();

print_mess("Successful termination:\n");
print_mess("Total running time (hours):",int((($stop-$start)/3600)*1000)/1000," hours\n");
print_mess("Total running time (minutes):",int((($stop-$start)/60)*1000)/1000," mins\n");
print_mess("Total running time (seconds):",int(($stop-$start)*1000)/1000," secs");
print_ok();
exit(0);
##


############ Subroutines

sub print_mess
{
        my @mess = @_;

        print STDERR color("bold green"),"%%%% @mess" if (exists($opt{v}));
	print STDERR color("reset");
}

sub print_error
{
        my @mess = @_;

        print STDERR color("bold green"),"%%%% @mess\n";
	print STDERR color("reset");
	exit();
}

sub print_ok
{
    if (exists($opt{v}))
    {
	print STDERR color("bold green"), " [OK]\n\n";
	print STDERR color("reset");
    }
}

