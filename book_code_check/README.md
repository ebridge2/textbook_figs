# Book code checker utility

This is a utility that creates reproducible notebooks for each section of the book directly from the overleaf draft. Given the overleaf draft of the book at `../../draft-textbook/` (e.g., acquired via git clone), call:

```
bash code_check.sh
```

Which will create the reproducible notebooks by scraping all code from the book, and place them in the appropriate location of the jupyter book project.

Then, follow the instructions [here](https://github.com/ebridge2/textbook_figs/tree/main?tab=readme-ov-file#usage) to compile all of the figures for the book, to verify that everything compiles before pushing.
