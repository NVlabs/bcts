1. Set your local project folder to variable `export PROJECT_DIR=your_proj_dir_full_path`
1. Clone bcts project:
 ``git clone https://github.com/galdl/bcts.git $PROJECT_DIR``
2. Clone rainbow project, version 1.3:
``git clone -b 1.3 https://github.com/Kaixhin/Rainbow.git $PROJECT_DIR``
3. Create pretrained agents dir:
``mkdir $PROJECT_DIR/bcts/pretrained``
4. Download agent(s) of your choice into `$PROJECT_DIR/bcts/pretrained` from:
https://github.com/Kaixhin/Rainbow/releases/tag/1.3
6. Run using docker: `docker run -v $PROJECT_DIR:/mount ... `





 