#################################
#	Environment Requirements	#
#################################

Your code must be written in Python and function properly in a Conda environment with Python 3.11 and Jupyter preinstalled. An appropriate conda environment will be provided by the TAs. Activate a conda environment with `conda activate <insert environment name>` before use, if it is not already activated. 

Navigate to the directory of this repo. Start Jupyter using one of the command `jupyter notebook --no-browser` and then paste the generated link to your browser.

With Jupyter running, complete the notebook including the name of the assignment.

#################################
#	Submission Rules	#
#################################

Included in your repository is a script named `finalize.sh`, which you will use to indicate which version of your code is the one to be graded. When you are ready to submit your final version, run the command `./finalize.sh` from your local Git directory, then commit and push your code. The finalize script will create a text file, `readyToSubmit.txt`, that is populated with information in a known format for grading purposes. You may commit and push as much as you want, but your submission will be confirmed as final if `readyToSubmit.txt` exists and is populated with the text generated by `finalize.sh` at the submission deadline on the due date. If you do NOT plan to submit before the deadline, then you should NOT run the `finalize.sh` script until your final submission is ready. If you accidentally run `finalize.sh` before you are ready to submit, do not commit or push your repo and delete `readyToSubmit.txt`. Once your final submission is ready, run `finalize.sh`, commit and push your code, and do not make any further changes to it

Late submissions will be penalized in accordance with the syllabus or as otherwise communicated by the instructors or TAs.

If `finalize.sh` gives a permission error, try to run `chmod 755 finalize.sh` on the file to make it executable.

#################################
#	Git Cheat Sheet	#
#################################
* `git status` - view uncommited changes in local repo
* `git add *` - adds all un-ignored files to the staged commit
* `git commit -m "some message here"` - forms the commit with an in-line commit message
* `git push` - pushes your commit to your online repo
