#!/bin/bash

python scrape_book.py ../../draft-textbook/ --output_dir ./

# Appendix
mv appendix_appA_notebook.ipynb ../textbook/Appendix/AppA/
mv appendix_appB_notebook.ipynb ../textbook/Appendix/AppB/
mv appendix_appC_notebook.ipynb ../textbook/Appendix/AppC/

# Applications
mv applications_ch6_notebook.ipynb ../textbook/applications/Ch6/
mv applications_ch7_notebook.ipynb ../textbook/applications/Ch7/
mv applications_ch8_notebook.ipynb ../textbook/applications/Ch8/
mv applications_ch9_notebook.ipynb ../textbook/applications/Ch9/

# Representations
mv representations_ch3_notebook.ipynb ../textbook/representations/Ch3/
mv representations_ch4_notebook.ipynb ../textbook/representations/Ch4/
mv representations_ch5_notebook.ipynb ../textbook/representations/Ch5/

# Foundations
mv foundations_ch2_notebook.ipynb ../textbook/foundations/Ch2/ 
