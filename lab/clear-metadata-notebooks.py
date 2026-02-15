from pathlib import Path
from typing import cast

from nbconvert import NotebookExporter
from nbconvert.preprocessors import (ClearMetadataPreprocessor,
                                     ClearOutputPreprocessor)


"""Clear the metadata from all notebooks in subdirectories of the script."""
def main(clear_out=True, clear_meta=True)-> None:
    exporter = NotebookExporter()
    exporter.register_preprocessor(ClearOutputPreprocessor(), enabled=clear_out)
    clear_meta_preproc = ClearMetadataPreprocessor()
    clear_meta_preproc.preserve_cell_metadata_mask = {"tags"}
    exporter.register_preprocessor(clear_meta_preproc, enabled=clear_meta)
    own_path = Path(__file__)

    for path in own_path.parent.rglob('*.ipynb'):
        print(f"Start processing {path}")
        clean_notebook, _ = exporter.from_filename(cast(str, path))

        with open(path, "w", encoding="utf-8") as f:
            f.write(cast(str, clean_notebook))


if __name__ == "__main__":
    main()
