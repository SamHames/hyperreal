# Download and prepare stackoverflow data for analysis
#
# Requirements: 7zz, python
#
# This script will need around ~300GiB of hard drive space to run.
#

# Download archive dumps
mkdir data
wget -c -N -O data/Posts.7z \
	https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z
	
wget -c -N -O data/Comments.7z \
	https://archive.org/download/stackexchange/stackoverflow.com-Comments.7z
	
wget -c -N -O data/Users.7z \
	https://archive.org/download/stackexchange/stackoverflow.com-Users.7z

# Extract data
7zz x data/Posts.7z -odata
7zz x data/Comments.7z -odata
7zz x data/Users.7z -odata

# Setup a python environment for this experiment
python -m venv stackoverflow_analysis
source stackoverflow_analysis/bin/activate

pip install hyperreal[stackexchange]

hyperreal stackexchange-corpus add-site \
	data/Posts.xml \
	data/Comments.xml \
	data/Users.xml \
	https://stackoverflow.com \
	data/stackoverflow.db

hyperreal stackexchange-corpus index \
	--doc-batch-size 50000 \
	data/stackoverflow.db \
	data/stackoverflow_index.db

hyperreal model \
	--clusters 512 \
	--include-field Post \
	--include-field Tag \
	--min-docs 29 \
	--random-seed 2023 \
	--tolerance 0.01 \
	data/stackoverflow_index.db

# Launch the webserver, then navigate to localhost:8080 in your browser
hyperreal stackexchange-corpus serve \
	data/stackoverflow.db \
	data/stackoverflow_index.db


