# general config
bucc="bucc2018"
data="."
xdir=${data}/BUCC	# tar files as distrubuted by the BUCC evaluation
ddir=${data}/${bucc}	# raw texts of BUCC
edir=${data}/embed	# normalized texts and embeddings
langs=("fr" "de" "ru" "zh")
ltrg="en"		# English is always the 2nd language

# encoder
# model_dir="${LASER}/models"
# encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
# bpe_codes="${model_dir}/93langs.fcodes"


###################################################################
#
# Extract files with labels and texts from the BUCC corpus
#
###################################################################

GetData () {
  fn1=$1; fn2=$2; lang=$3
  outf="${edir}/${bucc}.${lang}-${ltrg}.${fn2}"
  for ll  in ${ltrg} ${lang} ; do
    inf="${ddir}/${fn1}.${ll}"
    if [ ! -f ${outf}.txt.${ll} ] ; then
      echo " - extract files ${outf} in ${ll}"
      cat ${inf} | cut -f1 > ${outf}.id.${ll}
      cat ${inf} | cut -f2 > ${outf}.txt.${ll}
    fi
  done
}

ExtractBUCC () {
  slang=$1
  tlang=${ltrg}

  pushd ${data} > /dev/null
  if [ ! -d ${ddir}/${slang}-${tlang} ] ; then
    for tf in ${xdir}/${bucc}-${slang}-${tlang}.*.tar.bz2 ; do
      echo " - extract from tar `basename ${tf}`"
      tar jxf $tf
    done
  fi

  GetData "${slang}-${tlang}/${slang}-${tlang}.sample" "dev" ${slang}
  GetData "${slang}-${tlang}/${slang}-${tlang}.training" "train" ${slang}
  GetData "${slang}-${tlang}/${slang}-${tlang}.test" "test" ${slang}
  popd > /dev/null
}

echo -e "\nProcessing BUCC data in ${data}"

# create output directories
for d in ${ddir} ${edir} ; do
  mkdir -p ${d}
done

for lsrc in ${langs[@]} ; do
  ExtractBUCC ${lsrc}
done
