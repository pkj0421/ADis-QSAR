import logging
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from rdkit.ML.Cluster import Butina
from rdkit.Chem import AllChem, DataStructs


# module info option
logger = logging.getLogger(__name__)

# ignore warning
pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=FutureWarning)


class Cluster:
    """
    Input data need 'ROMol' column created from 'Smiles' column
    Output: central structures, remain structures
    """

    def __init__(self):
        pass

    def run(self, data, clusters, rb, max_cpu):
        # select reference structures
        logger.info(f"Generating distance matrix")
        data = data.reset_index(drop=True)
        dists, nfps = self._fps(data, rb, max_cpu)

        if len(data) >= 10000:
            cutoff_rg = np.array_split(np.arange(0.0001, 1, 0.0001), 100)
            fpool = Pool(processes=max_cpu)
            tcuts = tqdm(
                [{'dists': dists, 'nfps': nfps, 'cutoff': x, 'clusters': clusters, 'lst': True} for x in cutoff_rg],
                desc='Calculate sim rough cutoff...')
            f_css = fpool.map(self._cutoff, tcuts)
            fpool.close()
            fpool.join()

            new_cfs = cutoff_rg[self._match(f_css, clusters)]
            tncfs = tqdm(
                [{'dists': dists, 'nfps': nfps, 'cutoff': x, 'clusters': clusters, 'lst': False} for x in new_cfs],
                desc='Calculate sim detail cutoff...')
            pool = Pool(processes=max_cpu)
            css = pool.map(self._cutoff, tncfs)
            pool.close()
            pool.join()

            cts = [ref[0] for ref in css[self._match(css, clusters)][0]]
            c_com = data[data.index.isin(cts)]
            r_com = data[~data.index.isin(cts)]
            print(f"center compounds : {len(c_com)} | remain compounds : {len(r_com)}")
            return c_com, r_com

        else:
            cs_lst = []
            for cutoff in tqdm(sorted(np.arange(0.01, 1, 0.01), reverse=True)):
                cf = round(cutoff, 5)
                cs = Butina.ClusterData(dists, nfps, cf, isDistData=True)
                cs_lst += [(cs, len(cs))]
                if len(cs) == clusters:
                    cts = [ref[0] for ref in cs]
                    c_com = data[data.index.isin(cts)]
                    r_com = data[~data.index.isin(cts)]
                    logger.info(f"center compounds : {len(c_com)} | remain compounds : {len(r_com)}")
                    return c_com, r_com
            cts = [ref[0] for ref in cs_lst[self._match(cs_lst, clusters)][0]]
            c_com = data[data.index.isin(cts)]
            r_com = data[~data.index.isin(cts)]
            logger.info(f"center compounds : {len(c_com)} | remain compounds : {len(r_com)}")
            return c_com, r_com

    def _fps(self, dt, rb, mx):
        rd, bt = rb
        bit_generator = AllChem.GetMorganFingerprintAsBitVect
        dt['bits'] = dt['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt))
        data_fps = dt['bits'].tolist()

        nfps = len(data_fps)
        tfps = tqdm([(data_fps, i) for i in range(1, nfps)], desc='Calculate distance matrix...')

        sim_pool = Pool(mx)
        sim_result = sim_pool.map(self._tanimoto, tfps)
        sim_pool.close()
        sim_pool.join()

        dists = []
        sorted_sims = sorted(sim_result, key=lambda x: x[0])
        [dists.extend(x[1]) for x in sorted_sims]
        return dists, nfps

    def _cutoff(self, params):
        lst = params['lst']
        cfs = params['cutoff']
        if lst:
            cs_lst = []
            for cutoff in cfs:
                cf = round(cutoff, 5)
                cs = Butina.ClusterData(params['dists'], params['nfps'], cf, isDistData=True)
                cs_lst += [(cs, len(cs))]
            result = cs_lst[self._match(cs_lst, params['clusters'])]
            return result
        else:
            cf = round(cfs, 5)
            cs = Butina.ClusterData(params['dists'], params['nfps'], cf, isDistData=True)
            return cs, len(cs)

    @staticmethod
    def _tanimoto(box):
        dps, idx = box
        sims = DataStructs.BulkTanimotoSimilarity(dps[idx], dps[:idx])
        fsims = [1 - x for x in sims]
        return idx, fsims

    @staticmethod
    def _match(lst, clusters):
        arr = np.array([a[1] for a in lst])
        diffs = np.abs(arr - clusters)
        min_index = np.argmin(diffs)
        return min_index

    @staticmethod
    def _single(dt, rb, cut_lst):
        rd, bt = rb
        bit_generator = AllChem.GetMorganFingerprintAsBitVect
        dt['bits'] = dt['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt))
        data_fps = dt['bits'].tolist()

        dists = []
        nfps = len(data_fps)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(data_fps[i], data_fps[:i])
            dists.extend([1 - x for x in sims])

        css = [Butina.ClusterData(dists, nfps, i, isDistData=True) for i in cut_lst]
        css_result = [[c, len(c)] for c in css if c]
        return css_result


class Vector:

    def __init__(self):
        pass

    def run(self, g1, g2, n_cores, ss=True):

        # low memory
        if 'Active' in g2.columns:
            need_cols = ['Compound_ID', 'bits', 'Active']
            exist_act = True
        else:
            need_cols = ['Compound_ID', 'bits']
            exist_act = False

        g1_start = g1[need_cols]
        g2_start = g2[need_cols]

        # pair
        pool = Pool(processes=n_cores)
        if ss:
            if exist_act:
                func = partial(self._compare, ref=g1_start)
            else:
                func = partial(self._compare, ref=g1_start, ext=True)
        else:
            if exist_act:
                func = partial(self._compare, ref=g1_start, ss=False)
            else:
                func = partial(self._compare, ref=g1_start, ext=True, ss=False)

        g2_progress = tqdm([row[1] for row in g2_start.iterrows()], total=len(g2_start))
        vectors = pd.concat(pool.map(func, g2_progress)).reset_index(drop=True)

        pool.close()
        pool.join()

        return vectors

    def _compare(self, compound, ref, ext=False, ss=True):

        pairs = pd.DataFrame()

        compound_fps = np.array(compound['bits'])
        compound_fps_sc = np.where(compound_fps == 1, 2, compound_fps)
        keys = [f"f_{x}" for x in range(len(compound_fps_sc))]

        for _, ref_row in ref.iterrows():
            compare_fps = np.array(ref_row['bits'])
            compare_fps_sc = np.where(compare_fps == 1, 4, compare_fps)

            sum_fps_sc = compound_fps_sc + compare_fps_sc
            class_fps = np.array([self._class(xi) for xi in sum_fps_sc])

            fps_dict = dict(zip(keys, class_fps))
            pairs = pairs.append(fps_dict, ignore_index=True)

        if ss:
            vector = pairs.sum().to_frame().T
            vector.insert(0, 'Compound_ID', compound['Compound_ID'])

            if not ext:
                vector.insert(1, 'AD', compound['Active'])
        else:
            vector = pairs.copy()
            vector.insert(0, 'Compound_ID', compound['Compound_ID'])
            vector.insert(1, 'G1', ref['Compound_ID'])

            if not ext:
                vector.insert(2, 'AD', compound['Active'])
        return vector

    @staticmethod
    def _class(val):
        try:
            return val / 2
        except:
            return 0