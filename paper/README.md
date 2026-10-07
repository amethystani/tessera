# Paper

Overleaf-ready: zip this folder and set main.tex as the main document.
Compiler: pdfLaTeX. Bibliography: BibTeX (runs automatically, uses ref.bib).

main.tex            preamble, title, and the order of everything below
sections/           main text, one file per section
  00_abstract
  01_introduction
  02_related_work
  03_decomposition  section 3 header; pulls in 03a to 03d
    03a_identity
    03b_published_results
    03c_first_party
    03d_multilingual
  04_certificate
  05_calibration
  06_checkpoints
  07_conclusion
  08_limitations    unnumbered
  09_ethics         unnumbered
appendix/           appendices A to G, in order
tables/ figures/    generated from the results by experiments/make_paper_assets.py
                    in the code repo. Edit text in sections/, but change numbers
                    in tables and figures by regenerating them, not by hand.
ref.bib             bibliography
acl.sty, acl_natbib.bst   style files
