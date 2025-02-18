import warnings
from pathlib import Path
import subprocess
from rdkit import Chem
import nglview as nv
from openbabel import pybel
from MDAnalysis import Universe
import numpy as np
import redo
import pandas as pd
import os
import shutil
from pathlib import Path 


class Structure(Universe):
    """
    Core object to load structures with.

    Thin wrapper around MDAnalysis.Universe objects
    """

    @classmethod
    @redo.retriable(attempts=10, sleeptime=2)
    def from_pdbid(cls, pdbid):
        import mmtf

        return cls(mmtf.fetch(pdbid))

    @classmethod
    def from_string(cls, pdbid_or_path, **kwargs):
        import os

        if os.path.isfile(pdbid_or_path):
            return cls(pdbid_or_path, **kwargs)
        return cls.from_pdbid(pdbid_or_path, **kwargs)

    @classmethod
    def from_atomgroup(cls, *args):
        """Create a new new :class:`Structure` from one or more
        ``AtomGroup`` instances.

        Parameters
        ----------
        *args : ``AtomGroup``
            One or more AtomGroups.

        Returns
        -------
        structure : :class:`Structure`

        Raises
        ------
        ValueError
            Too few arguments or an AtomGroup is empty and
        TypeError
            Arguments are not :class:`AtomGroup` instances.

        Notes
        -----
        This is take straight from ``MDAnalysis.universe``. Refer to that
        module for more information.

        """
        from MDAnalysis.coordinates.memory import MemoryReader
        from MDAnalysis.topology.base import squash_by
        from MDAnalysis.core import groups
        from MDAnalysis.core.topology import Topology
        from MDAnalysis.core.topologyattrs import AtomAttr, ResidueAttr, SegmentAttr

        if len(args) == 0:
            raise ValueError("Need at least one AtomGroup for merging")

        for a in args:
            if not isinstance(a, groups.AtomGroup):
                raise TypeError(repr(a) + " is not an AtomGroup")
        for a in args:
            if len(a) == 0:
                raise ValueError("cannot merge empty AtomGroup")

        # Create a new topology using the intersection of topology attributes
        blank_topology_attrs = set(dir(Topology(attrs=[])))
        common_attrs = set.intersection(*[set(dir(ag.universe._topology)) for ag in args])
        tops = set(["bonds", "angles", "dihedrals", "impropers"])

        attrs = []

        # Create set of attributes which are array-valued and can be simply
        # concatenated together
        common_array_attrs = common_attrs - blank_topology_attrs - tops
        # Build up array-valued topology attributes including only attributes
        # that all arguments' universes have
        for attrname in common_array_attrs:
            for ag in args:
                attr = getattr(ag, attrname)
                attr_class = type(getattr(ag.universe._topology, attrname))
                if issubclass(attr_class, AtomAttr):
                    pass
                elif issubclass(attr_class, ResidueAttr):
                    attr = getattr(ag.residues, attrname)
                elif issubclass(attr_class, SegmentAttr):
                    attr = getattr(ag.segments, attrname)
                else:
                    raise NotImplementedError(
                        "Don't know how to handle"
                        " TopologyAttr not subclassed"
                        " from AtomAttr, ResidueAttr,"
                        " or SegmentAttr."
                    )
                if type(attr) != np.ndarray:
                    raise TypeError(
                        "Encountered unexpected topology "
                        "attribute of type {}".format(type(attr))
                    )
                try:
                    attr_array.extend(attr)
                except NameError:
                    attr_array = list(attr)
            attrs.append(attr_class(np.array(attr_array, dtype=attr.dtype)))
            del attr_array

        # Build up topology groups including only those that all arguments'
        # universes have
        for t in tops & common_attrs:
            offset = 0
            bondidx = []
            types = []
            
            for ag in args:
                # create a mapping scheme for this atomgroup
                mapping = {a.index: i for i, a in enumerate(ag, start=offset)}
                offset += len(ag)

                tg = getattr(ag, t)
                bonds_class = type(getattr(ag.universe._topology, t))
                # Create a topology group of only bonds that are within this ag
                # ie we don't want bonds that extend out of the atomgroup
                tg = tg.atomgroup_intersection(ag, strict=True)

                # Map them so they refer to our new indices
                new_idx = [tuple([mapping[x] for x in entry]) for entry in tg.indices]
                bondidx.extend(new_idx)
                if hasattr(tg, "_bondtypes"):
                    types.extend(tg._bondtypes)
                else:
                    types.extend([None] * len(tg))


            if any(t is None for t in types):
                attrs.append(bonds_class(bondidx))
            
            else:
                types = np.array(types, dtype="|S8")
                attrs.append(bonds_class(bondidx, types))

        # Renumber residue and segment indices
        n_atoms = sum([len(ag) for ag in args])
        residx = []
        segidx = []
        res_offset = 0
        seg_offset = 0
        for ag in args:
            # create a mapping scheme for this atomgroup's parents
            res_mapping = {r.resindex: i for i, r in enumerate(ag.residues, start=res_offset)}
            seg_mapping = {r.segindex: i for i, r in enumerate(ag.segments, start=seg_offset)}
            res_offset += len(ag.residues)
            seg_offset += len(ag.segments)

            # Map them so they refer to our new indices
            residx.extend([res_mapping[x] for x in ag.resindices])
            segidx.extend([seg_mapping[x] for x in ag.segindices])

        residx = np.array(residx, dtype=np.int32)
        segidx = np.array(segidx, dtype=np.int32)

        _, _, [segidx] = squash_by(residx, segidx)

        n_residues = len(set(residx))
        n_segments = len(set(segidx))

        top = Topology(
            n_atoms,
            n_residues,
            n_segments,
            attrs=attrs,
            atom_resindex=residx,
            residue_segindex=segidx,
        )

        # Create and populate a universe
        coords = np.vstack([a.positions for a in args])
        u = cls(top, coords[None, :, :], format=MemoryReader)

        return u

    def write(self, *args, **kwargs):
        # Workaround for https://github.com/MDAnalysis/mdanalysis/issues/2679
        if self.dimensions is None:
            self.trajectory.ts._unitcell = np.zeros(6)
        return self.atoms.write(*args, **kwargs)
    
warnings.filterwarnings("ignore")
ob_log_handler = pybel.ob.OBMessageHandler()
pybel.ob.obErrorLog.SetOutputLevel(0)

pdb_id = "5C1M"
def write_pdb_from_id(identifier: str, file_name: str) -> None:
    """
    Download a structure from PDB and writes it to a PDB file.

    Parameters
    ----------
    identifier: str
        PDB identifier.
    file_name: str
        name of the output file.
    """

    structure = Structure.from_pdbid(identifier)
    # element information maybe missing, but important for subsequent PDBQT conversion
    if not hasattr(structure.atoms, "elements"):
        structure.add_TopologyAttr("elements", structure.atoms.types)

    # write the protein file to disk
    protein = structure.select_atoms("protein")
    protein.write(file_name)

def pdb_to_pdbqt(pdb_path, pdbqt_path, pH=7.4) -> None:
    """
    Convert a PDB file to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    pdb_path: str or pathlib.Path
        Path to input PDB file.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    molecule = list(pybel.readfile("pdb", str(pdb_path)))[0]
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)

def smiles_to_pdbqt(smiles, pdbqt_path, pH=7.4) -> None:
    """
    Convert a SMILES string to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    smiles: str
        SMILES string.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    molecule = pybel.readstring("smi", smiles)
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # generate 3D coordinates
    molecule.make3D(forcefield="mmff94s", steps=10000)
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)


def run_smina(
    ligand_path, protein_path, out_path, exhaustiveness=8
):
    """
    Perform docking with Smina.

    Parameters
    ----------
    ligand_path: str or pathlib.Path
        Path to ligand PDBQT file that should be docked.
    protein_path: str or pathlib.Path
        Path to protein PDBQT file that should be docked to.
    out_path: str or pathlib.Path
        Path to which docking poses should be saved, SDF or PDB format.
    pocket_center: iterable of float or int
        Coordinates defining the center of the binding site.
    pocket_size: iterable of float or int
        Lengths of edges defining the binding site.
    num_poses: int
        Maximum number of poses to generate.
    exhaustiveness: int
        Accuracy of docking calculations.


    Returns
    -------
    output_text: str
        The output of the Smina calculation.
    """
    cwd = os.getcwd()
    
    ligand_name = ligand_path.strip(".pdbqt") + '_results'
    new_dir = Path(f"{cwd}/{ligand_name}")
    
    os.mkdir(new_dir)

    shutil.move(ligand_path, new_dir)
    shutil.copy(protein_path, new_dir)
    
    new_ligand_path = Path(f"{new_dir}/{ligand_path}")
    new_protein_path = Path(f"{new_dir}/{protein_path}")
    
    logfile = Path(f"{new_dir}/logfile.txt")

    os.chdir(ligand_name)
    output_text = subprocess.check_output(
        [
            "smina",
            "--ligand",
            str(new_ligand_path),
            "--receptor",
            str(new_protein_path),
            "--out",
            str(Path(f"{new_dir}/{out_path}")),
            "--autobox_ligand",
            str(ligand_path),
            "--autobox_add",
            str(8),
            "--exhaustiveness",
            str(exhaustiveness),
        ],
        universal_newlines=True,  # needed to capture output text
    )
    with open(logfile, "w") as f:
        f.write(output_text)


    os.chdir(cwd)


write_pdb_from_id(pdb_id, "mu_receptor.pdb")
pdb_to_pdbqt("mu_receptor.pdb", "mu_receptor.pdbqt")

ligand_agonist ='C[C@@]12C[C@@]34C=C[C@@]1([C@H]([C@@]35CCN([C@@H]4Cc6c5cc(cc6)O)C)N[C@@H]2c7ccccc7)OC'

# smi_mol = Chem.MolFromSmiles(ligand_agonist)
cwd = os.getcwd()

expansion_df = pd.read_csv('new_query.csv')
for i in range(len(expansion_df)): 
    ligand = expansion_df.iloc[i]['smiles']
    ligand_name = expansion_df.iloc[i]['CIDs']

    smiles_to_pdbqt(ligand, f"{ligand_name}.pdbqt")
    try:
        run_smina(f"{ligand_name}.pdbqt", "mu_receptor.pdbqt",f"{ligand_name}_docking_poses.sdf")
    
    except:
        os.chdir(cwd)
       
        print(f'could not dock ligand {ligand_name}')

for i in range(len(expansion_df)):    
    ligand_name = expansion_df.iloc[i]['CIDs']
    os.chdir(f"{cwd}/{ligand_name}_results")

    with open(f"{ligand_name}_docking_poses.sdf", "r") as f:
        lines = f.readlines()
        if len(lines) > 0:

            for i in range(len(lines)):
                if lines[i].startswith('>  <minimizedAffinity>'):
                    affinity = float(lines[i+1].strip())

                    expansion_df.loc[expansion_df['CIDs'] == ligand_name, 'affinity'] = affinity
                    break

            os.chdir(cwd)

        else:
            expansion_df.loc[expansion_df['CIDs'] == ligand_name, 'affinity'] = 'NaN'
            os.chdir(cwd)



expansion_df.to_csv('expansion_results.csv', index=False)