import time
import pickle
import numpy as np
from dataclasses import dataclass
import RNA
import os


@dataclass
class Results:
    """Store marginal and conditional (un)paired probabilities + structures.
    Results from local folding are expectation values rather than probabilities.
    """

    window_size: int
    max_bp_span: int
    num_samples: int
    sequence: str
    seq_id: str
    unpaired_P: float = np.nan  # (marginal) unpaired probability per position
    paired_P: float = np.nan  # (marginal) unpaired probability per position
    cond_unpaired_P: float = (
        np.nan
    )  # conditional unpaired probability of k(1st index) given l(2nd index) is unpaired
    cond_paired_P: float = (
        np.nan
    )  # conditional paired probability (indices analogouse to cond_unpaired_P)
    joint_unpaired_P: float = (
        np.nan
    )  # joint unpaired probability per index/position pair
    joint_paired_P: float = np.nan  # joint paired probability per index/position pair
    bp_P: float = np.nan  # base pair probability matrix
    mea_DB: str = ""  # mea structure as dotbraket string
    mea_E: float = np.nan  # mea energy in kcal/mol
    mfe_E: float = np.nan  # mfe energy in kcal/mol
    mfe_DB: str = ""  # mfe structure as dotbraket string
    mu: float = np.nan  # pairwise mutual information


class PLfoldRevisited:
    """Class for computing (un)paired probabilities from local and global folding."""

    def __init__(
        self,
        seq_id="my_random_id",
        sequence=RNA.random_string(300, "ACGU"),
        window_size=200,
        num_samples=20,
        max_bp_span=150,
        data_dir="data",
        recalculate=False,
    ):

        self.seq_id = seq_id
        self.sequence = sequence
        self.window_size = window_size
        self.num_samples = num_samples
        self.max_bp_span = max_bp_span
        self.data_dir = data_dir
        self.recalculate = recalculate

        self.global_folding = Results(
            window_size=self.window_size,
            max_bp_span=self.max_bp_span,
            num_samples=self.num_samples,
            sequence=self.sequence,
            seq_id=self.seq_id,
        )
        self.local_exact = Results(
            window_size=self.window_size,
            max_bp_span=self.max_bp_span,
            num_samples=self.num_samples,
            sequence=self.sequence,
            seq_id=self.seq_id,
        )
        self.local_sampling = Results(
            window_size=self.window_size,
            max_bp_span=self.max_bp_span,
            num_samples=self.num_samples,
            sequence=self.sequence,
            seq_id=self.seq_id,
        )

        self.descriptor = "_".join(
            [
                seq_id,
                str(max_bp_span).zfill(4),
                str(window_size).zfill(4),
                str(num_samples).zfill(3),
                str(len(sequence)).zfill(5),
            ]
        )

        # Updating the unpaired probabilities for both local models
        self._update_local_folding()
        self._update_global_folding()

    def __get_h_k(self):
        """
        Return an array with numbers of windows that contain position k.
        """
        seq_len = len(self.sequence)
        res = []
        for k in range(1, seq_len + 1):

            if k < self.window_size and k < seq_len + 1 - self.window_size:
                res.append(k)
            elif k < self.window_size and k >= seq_len + 1 - self.window_size:
                res.append(seq_len + 1 - self.window_size)
            elif k >= self.window_size and k <= seq_len + 1 - self.window_size:
                res.append(self.window_size)
            elif k > seq_len + 1 - self.window_size:
                res.append(seq_len + 1 - k)

        return res

    def __get_h_kl(self):
        """
        Return matrix with numbers of windows that cover positions i and j.
        i: row index; corresponding to zero based position in sequence
        j: column index;  corresponding to zero based position in sequence
        """

        if len(self.sequence) == self.window_size:
            return np.ones((len(self.sequence), len(self.sequence)))

        res = np.full((len(self.sequence), len(self.sequence)), np.nan)

        for i in range(1, len(self.sequence) + 1):
            for j in range(i + 1, len(self.sequence) + 1):
                if j > i + self.window_size - 1:
                    break

                if (
                    i > len(self.sequence) - self.window_size + 1
                    and j <= self.window_size
                ):
                    res[i - 1][j - 1] = len(self.sequence) - self.window_size + 1

                elif (
                    j >= self.window_size
                    and i <= len(self.sequence) - self.window_size + 1
                ):
                    res[i - 1][j - 1] = self.window_size - (j - i)

                elif j <= self.window_size:
                    res[i - 1][j - 1] = i

                elif i > len(self.sequence) + 1 - self.window_size:
                    res[i - 1][j - 1] = len(self.sequence) + 1 - j
        return res

    def _pairing_probabilities_from_local_sampling(self):
        """
        Calculate marginal and joint probability of a base being (un)paired;
        and compute base pair probabilities (expectation values) from local folding sample.

        param sequence: nucleotide sequence
        param window_size: window size for local structure prediction
        param num_sample: number of structure (DB) samples per window

        return: arrays with the probability of being (un)paired per each base;
                matrices with conditional probabillity of being (un)paired;
                matrix with base pair probabilities;
        """
        t1 = time.perf_counter()

        def store_structure(s, data):
            """
            A simple callback function that stores a structure
            sample into a list
            """
            if s:
                data.append(RNA.ptable(s)[1:])

        # fold compound
        fc = self.fc_S

        count_unpaired = np.zeros(
            len(self.sequence)
        )  # for each pos, how many time it has been sampled unpaired
        count_paired = np.zeros(
            len(self.sequence)
        )  # for each pos, how many time it has been sampled paired
        joint_count_unpaired = np.zeros(
            (len(self.sequence), len(self.sequence))
        )  # How many time i is unpaired for the windows containing i and j
        cond_count_unpaired = np.zeros(
            (len(self.sequence), len(self.sequence))
        )  # How many time i is unpaired fgiven that j is unpaired
        window_count_independent = np.zeros((len(self.sequence), len(self.sequence)))
        window_count_total = np.zeros((len(self.sequence), len(self.sequence)))
        joint_count_paired = np.zeros(
            (len(self.sequence), len(self.sequence))
        )  # How many time i is paired for the windows containing i and j
        count_bps = np.zeros(
            (len(self.sequence), len(self.sequence))
        )  # matrix containing the count of paired(i,j) in the sample
        cond_count_paired = np.zeros(
            (len(self.sequence), len(self.sequence))
        )  # How many time i is paired fgiven that j is paired

        for window_start in range(0, len(self.sequence) - self.window_size + 1):
            """
            sample structures for the current window and store into wslist
            structures as pairing vectors
            (((...)).) = [10,8,7,0,0,0,3,2,0,1]
            """
            wslist = list()
            fc.pbacktrack_sub(
                self.num_samples,
                window_start + 1,
                window_start + self.window_size,
                store_structure,
                wslist,
            )
            wslist = np.asarray(wslist)

            # storing how many time each pos is unpaired
            unpaired = np.count_nonzero(wslist == 0, axis=0)

            # storing how many time each pos is paired
            paired = np.count_nonzero(wslist, axis=0)

            # sum unpaired for each position
            count_unpaired[window_start : window_start + self.window_size] += unpaired

            # sum paired for each position
            count_paired[window_start : window_start + self.window_size] += paired

            # create matrix containing number of time i and j are paired with each other
            for s in wslist:
                for k in range(0, len(s)):
                    if s[k] > k:
                        count_bps[window_start + k][window_start + s[k] - 1] += 1

            # transpose wslist (window structure list)
            # each row correspond to the pairing at one position (0=unpaired, number=pos of pairing partner)
            tr_wsl = np.transpose(wslist)

            for k in range(0, len(tr_wsl)):
                # range of all potential pairing partners of k
                # ============================================
                left_start = window_start + k - self.window_size + 1
                if left_start < 0:
                    left_start = 0
                right_end = window_start + k + self.window_size
                if right_end > len(self.sequence):
                    right_end = len(self.sequence)

                # window coverage per pair of positions
                # =====================================
                window_count_independent[window_start + k][
                    window_start + self.window_size : right_end
                ] += 1
                window_count_independent[window_start + k][left_start:window_start] += 1

                window_count_total[window_start + k][left_start:right_end] += 1

                # joint unpaired probabillity
                # ==================================
                # create a matrix with the same array in each row (corresponding to pos 'k' of the sequence)
                check = np.tile(tr_wsl[k], (len(tr_wsl), 1))

                # mask is the comparison of all pos (tr_wsl) with pos k (check)
                mask = tr_wsl + check

                # counting the 0 in mask per row = count of i and j unpaired in the sample
                joint_kl_unpaired_count = np.count_nonzero(mask == 0, axis=1)

                # matrix containing the number of structures with k and l unpaired
                joint_count_unpaired[window_start + k][
                    window_start : window_start + self.window_size
                ] += joint_kl_unpaired_count

                # conditional unpaired probabillity
                # ==================================

                # conditional probability of k being unapired given that l is unpaired
                conditional_kl_unpaired = joint_kl_unpaired_count / unpaired

                cond_count_unpaired[window_start + k][
                    window_start : window_start + self.window_size
                ] += conditional_kl_unpaired

                # unpaired probability of position k
                unpaired_k_P = unpaired[k] / self.num_samples

                cond_count_unpaired[window_start + k][
                    left_start:window_start
                ] += unpaired_k_P

                cond_count_unpaired[window_start + k][
                    window_start + self.window_size : right_end
                ] += unpaired_k_P

                # joint paired probabillity
                # ==================================

                # count of the number of samples in which k and j are paired (not necessarily with one another)
                n_pair_test = tr_wsl * tr_wsl[k]
                joint_kl_paired_count = np.count_nonzero(n_pair_test, axis=1)

                # matrix containing the sum of unpaired (i,j)
                joint_count_paired[window_start + k][
                    window_start : window_start + self.window_size
                ] += joint_kl_paired_count

                # conditional paired probabillity
                # ==================================
                # conditional probability of k being unapired given that l is unpaired
                conditional_kl_paired = joint_kl_paired_count / paired

                cond_count_paired[window_start + k][
                    window_start : window_start + self.window_size
                ] += conditional_kl_paired

                # paired probability of k
                paired_k_P = paired[k] / self.num_samples

                cond_count_paired[window_start + k][
                    left_start:window_start
                ] += paired_k_P

                cond_count_paired[window_start + k][
                    window_start + self.window_size : right_end
                ] += paired_k_P

        # array with numbers of intervals that contain position k
        h_k = self.__get_h_k()

        # (half) matrix with number of windows containing both pos k and l
        h_kl = self.__get_h_kl()

        # calculate unpaired probabilities
        unpaired_P = (count_unpaired / self.num_samples) / h_k

        # calculate paired probabilities
        paired_P = (count_paired / self.num_samples) / h_k

        # conditional probability k to be unpaired given l is unpaired
        cond_unpaired_P = cond_count_unpaired / window_count_total

        # joint probability of k and l being unpaired
        unpaired_mx = np.tile(unpaired_P, (len(unpaired_P), 1))
        joint_unpaired_P = (
            joint_count_unpaired / self.num_samples
            + window_count_independent * unpaired_mx * unpaired_mx.transpose()
        ) / np.transpose(np.tile(h_k, (len(h_k), 1)))

        # conditional probability k to be paired given l is paired
        cond_paired_P = cond_count_paired / window_count_total

        # joint probability that k and l are paired
        paired_mx = np.tile(paired_P, (len(paired_P), 1))
        joint_paired_P = (
            joint_count_paired / self.num_samples
            + window_count_independent * paired_mx * paired_mx.transpose()
        ) / np.transpose(np.tile(h_k, (len(h_k), 1)))

        # base pair probabilities
        bp_P = (count_bps / self.num_samples) / h_kl

        t2 = time.perf_counter()
        print(f"{t2 - t1:0.4f} seconds for local sampling")

        return (
            unpaired_P,
            paired_P,
            cond_unpaired_P,
            cond_paired_P,
            bp_P,
            joint_unpaired_P,
            joint_paired_P,
        )

    def _get_mea_from_local_sampling(self):
        def bpp_list(bpp):
            bpp_list = []
            for i in range(0, len(bpp) - 1):
                for j in range(i + 1, len(bpp)):
                    if np.isnan(bpp[i][j]) == False and bpp[i][j] > 0.2:
                        bpp_list.append(RNA.ep(i + 1, j + 1, bpp[i][j], type=0))
            return bpp_list

        def unp_list(unp):
            unp_list = []
            for i in range(0, len(unp)):
                if np.isnan(unp[i]) == False:
                    unp_list.append(RNA.ep(i + 1, i + 1, unp[i], type=6))
            return unp_list

        p_list = bpp_list(self.local_sampling.bp_P) + unp_list(
            self.local_sampling.unpaired_P
        )
        mea_struct, mea = RNA.MEA_from_plist(p_list, self.sequence)
        return mea_struct, mea

    def _get_unpaired_probabilities_from_exact_local_folding(self):
        """
        Return array of unpaired probabilities (expectation values) according
        to exact local folding (PLfold model).
        """
        t1 = time.perf_counter()

        def up_callback(v, v_size, i, maxsize, what, pl_data):
            if what & RNA.PROBS_WINDOW_UP:
                pl_data.append(v)

        pl_data = [[np.nan, np.nan]]

        md = RNA.md()
        md.max_bp_span = self.max_bp_span
        md.window_size = self.window_size

        fc = RNA.fold_compound(self.sequence, md, RNA.OPTION_WINDOW)

        fc.probs_window(1, RNA.PROBS_WINDOW_UP, up_callback, pl_data)
        unpaired = np.array(pl_data).T.tolist()[1]
        t2 = time.perf_counter()
        print(f"{t2 - t1:0.4f} seconds for RNAPLfold")
        return np.array(unpaired[1:])  # remove leading nan value from RNAlib output

    def _update_local_folding(self):
        """(Re-)compute (un)paired probabilities and mea structure with local models."""
        try:
            if self.recalculate:
                raise OSError
            (
                self.local_sampling.unpaired_P,
                self.local_sampling.paired_P,
                self.local_sampling.cond_unpaired_P,
                self.local_sampling.cond_paired_P,
                self.local_sampling.bp_P,
                self.local_sampling.joint_unpaired_P,
                self.local_sampling.joint_paired_P,
                self.local_sampling.mea_DB,
                self.local_sampling.mea_E,
            ) = self.get_data(sampling=True, exact=False, global_f=False)

            self.local_exact.unpaired_P = self.get_data(
                sampling=False, exact=True, global_f=False
            )

        except OSError:
            data = {}

            def f(status, data):
                if status == RNA.STATUS_MFE_PRE:
                    data["start_m"] = time.time()
                elif status == RNA.STATUS_MFE_POST:
                    data["end_m"] = time.time()
                elif status == RNA.STATUS_PF_PRE:
                    data["start_p"] = time.time()
                elif status == RNA.STATUS_PF_POST:
                    data["end_p"] = time.time()

            ## Fold compound for stochastic backtracing
            # create model details
            md_S = RNA.md()
            md_S.uniq_ML = 1
            md_S.max_bp_span = self.max_bp_span

            # fold compound
            fc_S = RNA.fold_compound(self.sequence, md_S)
            fc_S.add_callback(f)
            fc_S.add_auxdata(data)

            # rescale pt (needs mfe)
            (ss, mfe) = fc_S.mfe()
            self.local_sampling.mfe_DB = ss
            self.local_sampling.mfe_E = mfe
            self.mfe_ss = ss
            fc_S.exp_params_rescale(mfe)

            (pp, pf) = fc_S.pf()

            fc_S.add_callback(f)
            fc_S.add_auxdata(data)

            self.fc_S = fc_S
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)

            if "start_m" in data and "end_m" in data:
                print(
                    "Filling DP matrices for MFE took {:0.4f} seconds".format(
                        data["end_m"] - data["start_m"]
                    )
                )

            if "start_p" in data and "end_p" in data:
                print(
                    "Filling DP matrices for PF took {:0.4f} seconds".format(
                        data["end_p"] - data["start_p"]
                    )
                )
            (
                self.local_sampling.unpaired_P,
                self.local_sampling.paired_P,
                self.local_sampling.cond_unpaired_P,
                self.local_sampling.cond_paired_P,
                self.local_sampling.bp_P,
                self.local_sampling.joint_unpaired_P,
                self.local_sampling.joint_paired_P,
            ) = self._pairing_probabilities_from_local_sampling()

            (
                self.local_sampling.mea_DB,
                self.local_sampling.mea_E,
            ) = self._get_mea_from_local_sampling()
            self.local_exact.unpaired_P = (
                self._get_unpaired_probabilities_from_exact_local_folding()
            )
            self._store_data_local()

    def _update_global_folding(self):
        """
        Get global bpp and unpaired probability with RNAfold -p and max_bp_span.
        """
        try:
            if self.recalculate:
                raise OSError
            (
                self.global_folding.unpaired_P,
                self.global_folding.bp_P,
            ) = self.get_data(sampling=False, exact=False, global_f=True)

        except OSError:
            t1 = time.perf_counter()

            if "fc_S" not in globals():
                md = RNA.md()
                md.uniq_ML = 1
                md.max_bp_span = self.max_bp_span

                fc = RNA.fold_compound(self.sequence, md)
                (ss, mfe) = fc.mfe()
                fc.exp_params_rescale(mfe)
                (pp, pf) = fc.pf()

            else:
                fc = self.fc_S

            # get matrix with base pair probabilities
            mx = fc.bpp()
            # convert base pairing matrix to numpy 2D array (not sure if needed?)
            mx = np.asarray(mx)
            # get array with unpaired probabilities
            cols = mx.sum(axis=0)
            rows = mx.sum(axis=1)
            unpaired = 1 - cols - rows

            t2 = time.perf_counter()
            print(f"{t2 - t1:0.4f} seconds for RNAfold -p")
            self.global_folding.unpaired_P = unpaired
            self.global_folding.bp_P = mx
            self._store_data_global()

    def rescale(self, new_window_size, new_num_samples):
        """
        Public function for rescale of window_size and num_samples.
        Updates the probabilities for both local models.
        """
        if new_window_size != self.window_size:
            self.window_size = new_window_size
            self.num_samples = new_num_samples
            self._update_local_folding()
        elif new_num_samples != self.num_samples:
            self.num_samples = new_num_samples
            self.local_exact.unpaired_P = (
                self._get_unpaired_probabilities_from_exact_local_folding()
            )

    def _store_data_local(self):
        descriptor = self.descriptor
        pickle.dump(
            self.local_sampling.unpaired_P,
            open(
                os.path.join(
                    self.data_dir, "sampling_unpaired_prob-" + descriptor + ".bin"
                ),
                "wb",
            ),
        )
        pickle.dump(
            self.local_sampling.paired_P,
            open(
                os.path.join(
                    self.data_dir, "sampling_paired_prob-" + descriptor + ".bin"
                ),
                "wb",
            ),
        )
        pickle.dump(
            self.local_sampling.cond_unpaired_P,
            open(
                os.path.join(
                    self.data_dir,
                    "sampling_cond_unpaired_prob-" + descriptor + ".bin",
                ),
                "wb",
            ),
        )
        pickle.dump(
            self.local_sampling.cond_paired_P,
            open(
                os.path.join(
                    self.data_dir,
                    "sampling_cond_paired_prob-" + descriptor + ".bin",
                ),
                "wb",
            ),
        )
        pickle.dump(
            self.local_sampling.bp_P,
            open(
                os.path.join(self.data_dir, "sampling_bpp-" + descriptor + ".bin"),
                "wb",
            ),
        )
        pickle.dump(
            self.local_sampling.joint_unpaired_P,
            open(
                os.path.join(
                    self.data_dir,
                    "sampling_joint_unpaired_prob-" + descriptor + ".bin",
                ),
                "wb",
            ),
        )
        pickle.dump(
            self.local_sampling.joint_paired_P,
            open(
                os.path.join(
                    self.data_dir,
                    "sampling_joint_paired_prob-" + descriptor + ".bin",
                ),
                "wb",
            ),
        )
        pickle.dump(
            self.local_sampling.mea_DB,
            open(
                os.path.join(self.data_dir, "sampling_mea-" + descriptor + ".bin"),
                "wb",
            ),
        )
        pickle.dump(
            self.local_sampling.mea_E,
            open(
                os.path.join(self.data_dir, "sampling_mea_e-" + descriptor + ".bin"),
                "wb",
            ),
        )

        pickle.dump(
            self.local_exact.unpaired_P,
            open(
                os.path.join(
                    self.data_dir, "exact_folding_unpaired-" + descriptor + ".bin"
                ),
                "wb",
            ),
        )

    def _store_data_global(self):
        descriptor = self.descriptor
        pickle.dump(
            self.global_folding.unpaired_P,
            open(
                os.path.join(
                    self.data_dir, "global_folding_unpaired-" + descriptor + ".bin"
                ),
                "wb",
            ),
        )
        pickle.dump(
            self.global_folding.bp_P,
            open(
                os.path.join(
                    self.data_dir, "global_folding_bpp-" + descriptor + ".bin"
                ),
                "wb",
            ),
        )

    def get_data(self, sampling=True, exact=True, global_f=True):
        descriptor = self.descriptor

        if sampling:
            unpaired_prob_path = os.path.join(
                self.data_dir, "sampling_unpaired_prob-" + descriptor + ".bin"
            )
            if not os.path.exists(unpaired_prob_path):
                raise OSError("File doesn't exist")
            else:
                sampling_unpaired_P = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir,
                            "sampling_unpaired_prob-" + descriptor + ".bin",
                        ),
                        "rb",
                    )
                )

                sampling_paired_P = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir, "sampling_paired_prob-" + descriptor + ".bin"
                        ),
                        "rb",
                    )
                )
                sampling_cond_unpaired_P = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir,
                            "sampling_cond_unpaired_prob-" + descriptor + ".bin",
                        ),
                        "rb",
                    )
                )
                sampling_cond_paired_P = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir,
                            "sampling_cond_paired_prob-" + descriptor + ".bin",
                        ),
                        "rb",
                    )
                )

                sampling_bp_P = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir, "sampling_bpp-" + descriptor + ".bin"
                        ),
                        "rb",
                    )
                )
                sampling_joint_unpaired_P = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir,
                            "sampling_joint_unpaired_prob-" + descriptor + ".bin",
                        ),
                        "rb",
                    )
                )
                sampling_joint_paired_P = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir,
                            "sampling_joint_paired_prob-" + descriptor + ".bin",
                        ),
                        "rb",
                    )
                )

                sampling_mea_DB = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir, "sampling_mea-" + descriptor + ".bin"
                        ),
                        "rb",
                    )
                )
                sampling_mea_E = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir, "sampling_mea_e-" + descriptor + ".bin"
                        ),
                        "rb",
                    )
                )

                return (
                    sampling_unpaired_P,
                    sampling_paired_P,
                    sampling_cond_unpaired_P,
                    sampling_cond_paired_P,
                    sampling_bp_P,
                    sampling_joint_unpaired_P,
                    sampling_joint_paired_P,
                    sampling_mea_DB,
                    sampling_mea_E,
                )

        if exact:
            plfold_prob_path = os.path.join(
                self.data_dir, "exact_folding_unpaired-" + descriptor + ".bin"
            )
            if not os.path.exists(plfold_prob_path):
                raise OSError("File doesn't exist")
            else:
                exact_folding_unpaired = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir,
                            "exact_folding_unpaired-" + descriptor + ".bin",
                        ),
                        "rb",
                    )
                )
                return exact_folding_unpaired

        if global_f:
            rnafold_unpaired = os.path.join(
                self.data_dir, "global_folding_unpaired-" + descriptor + ".bin"
            )
            if not os.path.exists(rnafold_unpaired):
                raise OSError("File doesn't exist")
            else:
                global_folding_unpaired = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir,
                            "global_folding_unpaired-" + descriptor + ".bin",
                        ),
                        "rb",
                    )
                )
                global_folding_bpmx = pickle.load(
                    open(
                        os.path.join(
                            self.data_dir, "global_folding_bpp-" + descriptor + ".bin"
                        ),
                        "rb",
                    )
                )
                return global_folding_unpaired, global_folding_bpmx
