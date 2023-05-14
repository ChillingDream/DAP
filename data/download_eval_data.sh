#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
DIR=${1-$REPO/download}
echo $DIR
mkdir -p $DIR

# Helper function to download the UD-POS data.
# In order to ensure backwards compatibility with the XTREME evaluation,
# languages in XTREME use the UD version used in the original paper; for the new
# languages in XTREME-R, we use a more recent UD version.

function download_tatoeba {
    base_dir=$DIR/tatoeba-tmp
    wget https://github.com/facebookresearch/LASER/archive/main.zip
    unzip -qq -o main.zip -d $base_dir/
    python data/gen_tatoeba_pair.py --src_dir $base_dir/LASER-main/data/tatoeba/v1 --tgt_dir $DIR/laser_tatoeba
    rm -rf $base_dir main.zip
    echo "Successfully downloaded data at $DIR/tatoeba" >> $DIR/download.log
}

function download_bucc18 {
    base_dir=$DIR/bucc2018/
    cd $DIR
    for lg in zh ru de fr; do
        wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.training-gold.tar.bz2 -q --show-progress
        tar -xjf bucc2018-${lg}-en.training-gold.tar.bz2
        wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.sample-gold.tar.bz2 -q --show-progress
        tar -xjf bucc2018-${lg}-en.sample-gold.tar.bz2
    done
    cd $REPO
    mv $base_dir/*/* $base_dir/
    for f in $base_dir/*training*; do mv $f ${f/training/test}; done
    for f in $base_dir/*sample*; do mv $f ${f/sample/dev}; done
    rm -rf $DIR/bucc2018*tar.bz2 $base_dir/{zh,ru,de,fr}-en/
    echo "Successfully downloaded data at $DIR/bucc2018" >> $DIR/download.log
}

download_bucc18
download_tatoeba