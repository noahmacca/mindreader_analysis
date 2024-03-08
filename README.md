# MindReader Analysis
For python scripts to explore and transform the data for the mindreader project, a web interface to interpret models like CLIP

How it works
Updated 3/8/24
- Run `create_db.py`
- Run `seed_db_to_files.py`. This processes the parquet files and adds them to a text file, for later writing to the db.
- Run `file_to_db.py`. This takes the text file and quickly writes to db. It's helpful to separate these steps because the seed_db_to_files is really slow and can't fit everything in memory, leading to flakiness.
- ALSO run `aws s3 sync ./s3_outputs_1 s3://mindreader-web` to sync the image files to s3 for serving to the webpage. This can take a while but be patient!