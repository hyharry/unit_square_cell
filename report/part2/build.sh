latex-clean

python file_trimmer.py t

pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex

python file_trimmer.py r

latex-clean
