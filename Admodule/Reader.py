import logging
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from pathlib import Path
from Admodule.Utils import save
from rdkit.Chem import PandasTools, SDMolSupplier


# module info option
logger = logging.getLogger(__name__)


class ChEMBL_reader:

    def __init__(self):
        pass

    def run(self, file_type, criteria, path_data, inhit=True, output=False):

        if not output:
            # read data
            return self._read(path_data, file_type)

        else:
            # load raw data
            raw_data = self._read(path_data, file_type)
            logger.info(f"Raw data counts : {len(raw_data)}")

            # filter
            ft_data = self._filter(raw_data)

            # determine activity
            final_data = self._duple(ft_data)
            final_data.rename(columns={'Molecule ChEMBL ID': 'Compound_ID'}, inplace=True)

            # divide
            if inhit:
                active, inactive = self._divide(final_data, criteria, inhits=True)
                logger.info(f"[IC50, Ki, Kd]\tActive range : x <= {criteria['act']}nM\tInactive range : {criteria['inact']}nM >= x")
                logger.info(f"[Inhibition%]\tActive range : {criteria['i-act']}% >= x\tInactive range : x <= {criteria['i-inact']}%")
            else:
                active, inactive = self._divide(final_data, criteria)
                logger.info(f"[IC50, Ki, Kd]\tActive range : x <= {criteria['act']}nM\tInactive range : {criteria['inact']}nM >= x")
            logger.info(f"Divide set : active ({len(active)}), inactive ({len(inactive)})")

            # save
            save(final_data.drop(columns='ROMol'), output, custom='total')
            save(active.drop(columns='ROMol'), output, custom='active')
            save(inactive.drop(columns='ROMol'), output, custom='inactive')
            return pd.concat([active, inactive])

    @staticmethod
    def _read(file, tp):

        if tp == '.tsv':
            molecules = pd.read_csv(file, sep='\t')

        elif tp == '.xlsx':
            molecules = pd.read_excel(file)

        elif tp == '.sdf':

            try:
                molecules = PandasTools.LoadSDF(file, smilesName='Smiles')

            except:
                molecules = pd.DataFrame()
                supple = SDMolSupplier(file)

                for mol in supple:
                    if mol:
                        add = [mol.Getprop('Molecule ChEMBL ID'), Chem.MolToSmiles(mol), mol, mol.Getprop('Active')]
                        add_series = pd.Series(add, index=['Molecule ChEMBL ID', 'Smiles', 'ROMol', 'Active'])
                        molecules = molecules.append(add_series, ignore_index=True)

        else:
            logger.error(f"'{tp}' is an invalid file.")

        total_cols = molecules.columns
        if 'Molecule ChEMBL ID' not in total_cols:
            return logger.error('Molecule ChEMBL ID column does not exist')
        if 'Smiles' not in total_cols:
            return logger.error('Smiles column does not exist')
        return molecules

    @staticmethod
    def _filter(raw_data):
        logger.info('Filtering raw data...')

        # remove not necessary columns
        select_cols = ['Molecule ChEMBL ID', 'Molecule Name', 'Molecular Weight',
                       'AlogP', 'Smiles', 'Standard Type', 'Standard Relation',
                       'Standard Units', 'Standard Value', 'Assay ChEMBL ID',
                       'Assay Description', 'BAO Format ID', 'BAO Label',
                       'Assay Organism', 'Target ChEMBL ID', 'Target Name',
                       'Target Organism', 'Target Type', 'Document ChEMBL ID',
                       'Source Description', 'Document Journal', 'Document Year']

        select_data = raw_data[select_cols]

        # remove missing values
        select_data.dropna(subset=['Smiles', 'Standard Value'], how='any', inplace=True)
        logger.info(f'Remove missing values : {len(raw_data) - len(select_data)}')

        # remove salts
        select_data['Smiles'] = select_data['Smiles'].apply(lambda x: max(x.split('.'), key=len))
        logger.info(f'Remove salts...')

        # generate ROMol
        PandasTools.AddMoleculeColumnToFrame(select_data)
        logger.info(f'Generate ROMol column...')

        # remove macro cycles
        select_data['macro'] = select_data['ROMol'].apply(lambda x: x.HasSubstructMatch(Chem.MolFromSmarts("[r{8-}]")))
        rm_macro = select_data[select_data['macro'] == False].drop(columns='macro')
        logger.info(f"Remove macro cycles : {len(select_data) - len(rm_macro)}")

        # select BAO Label 'single protein format'
        bao_filter = rm_macro[rm_macro['BAO Label'] == 'single protein format']
        logger.info(f"Select 'single protein format' data : {len(bao_filter)}")

        # pick activity types
        pick_type = [tp for tp in bao_filter['Standard Type'].unique() if tp in ['IC50', 'Ki', 'Kd', 'Inhibition']]
        type_filter = bao_filter[bao_filter['Standard Type'].isin(pick_type)]
        logger.info(f"Filtered activity type ['IC50', 'Ki', 'Kd', '%Inhibition'] : {len(type_filter)}")

        # select '=' Standard Relation
        rel_filter = type_filter[type_filter['Standard Relation'] == "'='"]
        logger.info(f"Filtered activity relation ['='] : {len(rel_filter)}")

        # unify units
        unit_filter = rel_filter[(rel_filter['Standard Units'] == '%') | (rel_filter['Standard Units'] == 'nM')]
        logger.info(f"Filtered activity units ['nM', '%'] : {len(unit_filter)}")
        return unit_filter

    @staticmethod
    def _determine(tp, vals):
        max = vals['Standard Value'].max()
        min = vals['Standard Value'].min()

        if not tp == 'inhit':

            if not max - min >= 100:
                vals.iloc[0, vals.columns.get_loc('Standard Value')] = round(vals['Standard Value'].mean(), 2)
                vals = vals[:1]
                return vals
            else:
                return pd.DataFrame()

        else:

            if not max - min >= 10:
                vals.iloc[0, vals.columns.get_loc('Standard Value')] = round(vals['Standard Value'].mean(), 2)
                vals = vals[:1]
                return vals
            else:
                return pd.DataFrame()

    @staticmethod
    def _classification(tp, df, criteria):
        acts = df.copy()

        if tp == 'main':
            acts['Active'] = acts['Standard Value'].apply(lambda x: 1 if x <= criteria['act'] else (0 if x >= criteria['inact'] else None))
        if tp == 'inhit':
            acts['Active'] = acts['Standard Value'].apply(lambda x: 1 if x >= criteria['i-act'] else (0 if x <= criteria['i-inact'] else None))

        active = acts[acts['Active'] == 1]
        inactive = acts[acts['Active'] == 0]
        return active, inactive

    def _duple(self, ft_data):

        # check duple ChEMBL ID
        duple = ft_data['Molecule ChEMBL ID'].duplicated(keep=False)
        duple_groups = ft_data[duple].groupby('Molecule ChEMBL ID')
        logger.info(f"IDs with multiple activities counts : {len(duple_groups)}")

        duple_results = pd.DataFrame()
        logger.info('Determining activity for IDs with multiple activities...')
        for group_idx, group in duple_groups:

            # check group activity types & fix activities
            types = group.groupby("Standard Type")
            type_lst = types.groups.keys()

            if 'IC50' in type_lst:
                ic50 = types.get_group('IC50')
                fix_ic50 = self._determine('ic50', ic50)
                if not fix_ic50.empty:
                    duple_results = pd.concat([duple_results, fix_ic50])
                    continue

            elif 'Ki' in type_lst:
                ki = types.get_group('Ki')
                fix_ki = self._determine('ki', ki)
                if not fix_ki.empty:
                    duple_results = pd.concat([duple_results, fix_ki])
                    continue

            elif 'Kd' in type_lst:
                kd = types.get_group('Kd')
                fix_kd = self._determine('kd', kd)
                if not fix_kd.empty:
                    duple_results = pd.concat([duple_results, fix_kd])
                    continue

            elif 'Inhibition' in type_lst:
                inhit = types.get_group('Inhibition')
                fix_inhit = self._determine('inhit', inhit)
                if not fix_inhit.empty:
                    duple_results = pd.concat([duple_results, fix_inhit])

        duple_filter = pd.concat([ft_data[~duple], duple_results]).reset_index(drop=True)
        duple_filter.sort_values(by='Standard Type', axis=0)
        logger.info(f"ChEMBL data preprocessing complete : {len(duple_filter)}")
        return duple_filter

    def _divide(self, df, criteria, inhits=False):
        mains = df[df['Standard Type'] != 'Inhibition']
        main_active, main_inactive = self._classification('main', mains, criteria)

        if inhits:
            inhits = df[df['Standard Type'] == 'Inhibition']
            inhit_active, inhit_inactive = self._classification('inhit', inhits, criteria)
            active = pd.concat([main_active, inhit_active]).reset_index(drop=True)
            inactive = pd.concat([main_inactive, inhit_inactive]).reset_index(drop=True)
        else:
            active = main_active.copy()
            inactive = main_inactive.copy()
        return active, inactive


class Custom_reader:

    def __init__(self):
        pass

    def run(self, file_type, path_data, output=False):

        if not output:
            # read data
            raw_data = self._read(path_data, file_type)
            mv_data = raw_data.dropna(subset=['Smiles'])
            if len(raw_data) - len(mv_data) != 0:
                logger.info(f'Remove missing values : {len(raw_data) - len(mv_data)}')
            PandasTools.AddMoleculeColumnToFrame(mv_data)
            return mv_data

        else:
            # load raw data
            raw_data = self._read(path_data, file_type)
            logger.info(f"Raw data counts : {len(raw_data)}")

            # filter
            ft_data = self._filter(raw_data)

            # save
            save(ft_data, output, custom='total')

            # divide & save
            active, inactive = self._divide(ft_data)
            logger.info(f"Divide : active ({len(active)}), inactive ({len(inactive)})")

            save(active.drop(columns='ROMol'), output, custom='active')
            save(inactive.drop(columns='ROMol'), output, custom='inactive')

    @staticmethod
    def _read(file, tp):

        if tp == 'df':
            molecules = file.copy()

        elif tp == '.tsv':
            molecules = pd.read_csv(file, sep='\t')

        elif tp == '.xlsx':
            molecules = pd.read_excel(file)

        elif tp == '.sdf':

            try:
                molecules = PandasTools.LoadSDF(file, smilesName='Smiles')

            except:
                molecules = pd.DataFrame()
                supple = SDMolSupplier(file)

                for mol in tqdm(supple):
                    if mol:
                        add = [mol.Getprop('Compound_ID'), Chem.MolToSmiles(mol), mol, mol.Getprop('Active')]
                        add_series = pd.Series(add, index=['Compound_ID', 'Smiles', 'ROMol', 'Active'])
                        molecules = molecules.append(add_series, ignore_index=True)

        else:
            logger.error(f"'{tp}' is an invalid file.")

        total_cols = molecules.columns
        if 'Compound_ID' not in total_cols:
            logger.error('Compound ID column does not exist')
            return
        if 'Smiles' not in total_cols:
            logger.error('Smiles column does not exist')
            return
        if 'Active' not in total_cols:
            logger.error('active column does not exist')
            return
        return molecules

    @staticmethod
    def _filter(raw_data):
        logger.info('Filtering raw data...')

        # remove missing values
        mv_data = raw_data.dropna(subset=['Smiles', 'Active'], how='any')
        logger.info(f'Remove missing values : {len(raw_data) - len(mv_data)}')

        # remove salts
        mv_data['Smiles'] = mv_data['Smiles'].apply(lambda x: max(x.split('.'), key=len))
        logger.info(f'Remove salts...')

        # generate ROMol
        PandasTools.AddMoleculeColumnToFrame(mv_data)
        logger.info(f'Generate ROMol column...')

        return mv_data

    @staticmethod
    def _divide(df):
        df.dropna(columns='Active', inplace=True)
        df['Active'] = df['Active'].astype('int')
        active = df[df['Active'] == 1].reset_index(drop=True)
        inactive = df[df['Active'] == 0].reset_index(drop=True)
        return active, inactive

