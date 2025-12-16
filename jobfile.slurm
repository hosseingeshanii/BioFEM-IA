#!/bin/bash
##SBATCH --export=NONE
##SBATCH --get-user-env=L

#SBATCH --job-name=u_fi_n_fu_lr_1e-2_fbsm_0_n
#SBATCH --output=output_u_fi_n_fu_lr_1e-2_fbsm_0_n.log
#SBATCH --error=error.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=300G

# Load modules



module purge
source petsc_3_20.sh


# Read values from control.dat
subdir=$(grep -oP '(?<=-subdir )\S+' control.dat)
if [[ -z "$subdir" ]]; then
  echo "Error: -subdir not found in control.dat"
  exit 1
fi
mkdir -p ./"$subdir"
cp control.dat ./"$subdir"/

# Read parameters
constitutive_law=$(grep -oP '(?<=-ConstitutiveLawNonLinear )\d+' control.dat)
inverse=$(grep -oP '(?<=-inverse )\d+' control.dat)

if [[ -n "$constitutive_law" && -n "$inverse" && "$constitutive_law" -eq 1 && "$inverse" -eq 1 ]]; then
  cp lr_fung.txt ./"$subdir"/
  cp init_fung.txt ./"$subdir"/
fi


# env > env_slurm.txt
echo "SLURM_JOB_ID =" $SLURM_JOB_ID

jobstats &
my_job_command
more_job_commands
jobstats

# mpirun ./testt
mpirun   -mca pml ucx   -mca btl '^uct,ofi'   -mca mtl '^ofi'  ./testt
# if [ "$constitutive_law" -eq 1 ] && [ "$inverse" -eq 1 ]; then
#   # module load GCCcore/12.2.0
#   # module load gnuplot/5.4.6
#   # gnuplot res_plot.gp
#   # gnuplot n_x_plot.gp
#   # mv res.png ./"$subdir"/
#   # mv n_x.png ./"$subdir"/
#   # mv output.log ./"$subdir"/
# fi

