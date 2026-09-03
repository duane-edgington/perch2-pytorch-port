#!/usr/bin/env bash

# This script runs `sox` for resampling days in a given year/month.
# year and month are required. An optional day range can be passed as
# the 3rd and 4th arguments (start_day end_day); it defaults to 1-31.
# Examples:
#    ./new_32k_resample_sox.sh 2018 11          # whole month (days 1-31)
#    ./new_32k_resample_sox.sh 2018 5 2 2       # only May 2, 2018
#    ./new_32k_resample_sox.sh 2018 5 1 7       # May 1 through 7, 2018
# Each resample is launched in its own process.

set -ue

year=$1
month=$2
days=$(seq "${3:-1}" "${4:-31}")  # optional day range; defaults to full month

audio_base_dir="/mnt/PAM_Archive"

decimated_base_dir="/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz"
#decimated_base_dir="/home/duane/google-multispecies-whale-detection/local/PAM_Analysis/GoogleOrcaModel/resampled_32kHz"

days_line="$(echo "${days}" | tr '\n' ' ')"

in_dir=$(printf "%s/%04d/%02d" ${audio_base_dir} "${year}" "${month}")

out_dir=$(printf "%s/%04d/%02d" ${decimated_base_dir} "${year}" "${month}")
mkdir -p "${out_dir}"

printf "Starting resample_sox.sh: %04d-%02d days: %s\n" "${year}" "${month}" "${days_line}"

#use SoX to resample the audio data directly. 
# rate converts to kHz. the -v flag is for very high quality.
# convert to 16 bit depth high (required by the google model)
# highpass 10Hz (to remove dc offset)
# vol 3 (to adjust volume 3x, making the signal correct in Volts)
# fade logarithmic 0.1 sec fade in, -0 sec hold (i.e. full duration no matter how long), 0.1 sec fade out

for day in ${days}; do
  prefix=$(printf "%s/MARS_%04d%02d%02d" "${in_dir}" "${year}" "${month}" "${day}")
  #for infile in "${prefix}"_06*.wav; do
  for infile in "${prefix}"_*.wav; do
    basename=$(basename "${infile}" .wav)
    outfile="${out_dir}/${basename}_resampled_32kHz.wav"
    echo "infile = ${infile}"
    echo "outfile = ${outfile}"
    sox "${infile}" -b 16 "${outfile}" rate -v 32000 highpass 10 fade 0.1 -0 0.1 vol 3 &
  done

done
wait
