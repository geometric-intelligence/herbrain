import os
import sys
from pathlib import Path

from herbrain.pregnancy.visualization.paraview import update_visualization


def main(project_dir):
    result_dir = project_dir / 'meshes_nico'
    for structure in os.listdir(result_dir):
        if structure == 'results.csv':
            continue
        for config in os.listdir(result_dir / structure):
            if (result_dir / structure / config / 'visualisation_names.json').exists():
                print(structure, config)
                update_visualization(project_dir, structure, config)


if __name__ == '__main__':
    p_dir = Path(sys.argv[1])
    main(p_dir)
