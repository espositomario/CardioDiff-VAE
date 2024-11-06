#!/usr/bin/perl -w                                #
#                                                 #
# Script MapChIPseq.pl                            #
#                                                 #
#                                                 #
# Input : GENOME, ASSEMBLY, FASTQ file, NAME      # 
# Output: Mapping+BAM+UCSCprofile                 #
#                                                 #
#                                                 #
# by Enrique Blanco @ CRG (2021)                  #
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
my ($genome,$version,$fastq_file,$name);
my $n_files;
my $newdir;

(getopts('v',\%opt)) or print_error("parser: Problems reading options\n");

print_mess("MapChIPseq.pl by Enrique Blanco @ CRG (2018)\n\n");
print_mess("Stage 0.  Reading options");

$n_files = $#ARGV+1;
($n_files == 4) or print_error("USAGE: Four arguments are required but $n_files are provided!\nMapChIPseq.pl -v <GENOME> <VERSION> <FASTQ> <NAME>\n");
($genome,$version,$fastq_file,$name) = @ARGV;

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
	if ($genome eq "chicken")
	{
	    if ($version eq "galgal6")
	    {
		$GENOME = "\$GALGAL6";
		$MAPS = $MAPS."/galgal6";
	    }
	    else
	    {
		print_error("Please, use galgal6 as genome assembly for chicken ($version)");
	    }
	}
	else
	{
	    if ($genome eq "oppossum")
	    {
    		if ($version eq "mondom5")
		{
		    $GENOME = "\$MONDOM5";
		    $MAPS = $MAPS."/mondom5";
		}
		else
		{
		    print_error("Please, use galgal6 as genome assembly for oppossum ($version)");
		}
	    }
	    else
	    {
		if ($genome eq "fly")
		{
		    if ($version eq "dm3")
		    {
			$GENOME = "\$FLY3";
			$MAPS = $MAPS."/dm3";
		    }
		    else
		    {
			print_error("Please, use dm3 as genome assembly for fly ($version)");
		    }
		}
		else
		{
		    print_error("Please, select <human>, <mouse> <oppossum> <chicken> or <fly> for the genome mapping");
		}
	    }
	}
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


print_mess("Stage 1.  Mapping the ChIPseq sample ($fastq_file -> $MAPS/$name.sam)\n");

# Modified line for Bowtie2
$command = "bowtie2 -p $NT -x $GENOME/genome -U $fastq_file -S $MAPS/$name.sam 2> info/bowtie2_$name.txt";


print_mess("Running $command");
system($command);
print_ok();
##

## Step 2. Generate the BAM file from the SAM file by bowtie
my $flagstat_sample;

print_mess("Stage 2.  Generating the BAM file ($MAPS/$name.sam -> $MAPS/$name.bam)\n");
$command = "samtools view -b -q 10 -F 0x4 -o $MAPS/$name.bam $MAPS/$name.sam";
# -F 0x4: remove unmapped reads
# -q 10: remove reads with mapping quality lower than 10 less likely to be uniquely mapped
print_mess("Running $command");
system($command);
print_ok();

$command = "rm -f $MAPS/$name.sam";
print_mess("Running $command");
system($command);
print_ok();

$flagstat_sample = "$INFO/flagstat_$name.txt";
$command = "samtools flagstat $MAPS/$name.bam > $flagstat_sample";
print_mess("Running $command");
system($command);

$command = "cat $flagstat_sample";
print_mess("Running $command\n");
system($command);
print_ok();

##

## Step 3. Building the ChIPseq profile for UCSC
print_mess("Stage 3.  Building the normalized profile for UCSC ($MAPS/$name.bam -> $name)\n");
$command = "buildChIPprofile $GENOME/ChromInfo.txt $MAPS/$name.bam $name";
print_mess("Running $command");
system($command);
print_ok();
##

## Step 4. Finishing successful program execution
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

