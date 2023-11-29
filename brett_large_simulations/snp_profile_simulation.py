import argparse
import numpy as np
import random
import pandas as pd
from collections import defaultdict

class EmbryoSequenceSimulator:
    def __init__(self, input_df, n_embryos, avg_harvested_cells, sd_harvested_cells):
        
        self.input_df = input_df # cols: gene,chromosome,start,end,direction,sequence
        self.n_embryos = n_embryos
        self.avg_harvested_cells = avg_harvested_cells
        self.sd_harvested_cells = sd_harvested_cells

        self.alphabet = {"A":0, "C":1, "G":2, "T":3}
        self.genotypes = {
            "AA":1, "AC":2, "AG":3, "AT":4, 
            "CA":2, "CC":5, "CG":6, "CT":7,
            "GA":3, "GC":6, "GG":8, "GT":9, 
            "TA":4, "TC":7, "TG":9, "TT":10
        }
        
        # Global parental chromatid maps
        self.m0_chromatid = None
        self.m1_chromatid = None
        self.p0_chromatid = None
        self.p1_chromatid = None
        
    def global_seeding(self, heterozygosity):
        """
        Uniform seeding of SNP locations for each gene with frequency according to heterozygosity.
        Parental chromatid seeding (2 chromatids from mother, 2 chromatids from father)
        """
        # Setting SNP positions for each transcript in input_df
        snp_map = defaultdict(dict)
        sequence_map = defaultdict(dict)
        for idx,transcript_entry in self.input_df.iterrows():
            gene,chrom,start,end,direction,sequence = transcript_entry
            N = len(sequence)
            num_snps = round(N*heterozygosity)
            snp_positions = np.sort(random.sample(range(N),num_snps))
            snp_map[chrom][gene] = snp_positions
            sequence_map[chrom][gene] = sequence
        
        # Setting parental chromatid maps
        for parent in ["m0","m1","p0","p1"]:    
            chromatid = defaultdict(dict)
            for chrom,genes in snp_map.items():
                for gene,snp_positions in genes.items():
                    snp_profile = {locus:random.choice(list(self.alphabet.keys())) for locus in snp_positions}
                    sequence_list = [char for char in sequence_map[chrom][gene]]
                    for locus,char in snp_profile.items():
                        sequence_list[locus] = char
                    chromatid[chrom][gene] = "".join(sequence_list)
                
            # Assign seeded parental chromatid to its class field (global over all embryos)
            if parent == "m0":
                self.m0_chromatid = chromatid
            elif parent == "m1":
                self.m1_chromatid = chromatid
            elif parent == "p0":
                self.p0_chromatid = chromatid
            elif parent == "p1":
                self.p1_chromatid = chromatid
                
        return snp_map
    
    def embryo_population_simulation(self, heterozygosity, count_df=None):
        """
        Simulating SNP profile for each embryo across all transcript entries in input_df.
        For this simulator, not explicitly generating scRNA-seq reads for each embryo (reduces space)
        """
        # map: idx of transcript entry -> list of snp positions
        snp_map = self.global_seeding(heterozygosity)
        true_clusters = []
        cluster_label = 0
        population_profile = []
        for embryo in range(self.n_embryos):
            
            # each entry a list of genotypes for an embryo across all transcripts
            snp_profile,gene_locus_map = self.determine_embryo_genotype(snp_map)

            # Harvest some number of cells, which will theoretically have identical SNP profiles
            n_harvested_cells = round(np.random.normal(self.avg_harvested_cells,self.sd_harvested_cells))
            for __ in range(n_harvested_cells):
                if count_df is not None:
                    snp_profile_dropout = self.sample_dropout_profile(
                        snps=snp_profile.copy(),
                        loci_map=gene_locus_map,
                        count_df=count_df
                    )
                    population_profile.append(snp_profile_dropout)
                else:
                    population_profile.append(snp_profile)
                true_clusters.append(cluster_label)
            cluster_label += 1
                
        return np.array(population_profile),true_clusters,snp_map
    
    def sample_dropout_profile(self, snps, loci_map, count_df):
        """
        Sample a dropout profile from single-cell transcript read count df, and 
        introduce dropout over the snp_profile according to the dropout profile.
        """
        sampled_dprofile = random.choice(count_df.columns.tolist()[2:])
        for idx,row in count_df[["Lv_name",sampled_dprofile]].iterrows():
            gene,count = row
            if count == 0:
                for locus in loci_map[gene]:
                    snps[locus] = 0 # 0 genotype -> undetected
        
        return snps
        
    
    def determine_embryo_genotype(self, snp_map):
        """
        Simulate array containing genotypes over all transcripts 
        (concatenated into single list)
        """  
        # state (genotype) across all transcripts
        snp_genotypes,idx = [],0
        gene_locus_map = defaultdict(list)
                
        # map: idx of transcript entry -> sequence after homologous recombination
        m_chromatid,p_chromatid = self.simulate_embryo_sequence(snp_map)
        for chrom,genes in snp_map.items():
            for gene,snp_positions in genes.items():                
                m_sequence = m_chromatid[chrom][gene]
                p_sequence = p_chromatid[chrom][gene]
                for locus in snp_positions:
                    snp_genotype = "".join([m_sequence[locus],p_sequence[locus]])
                    snp_genotypes.append(self.genotypes[snp_genotype])
                    gene_locus_map[gene].append(idx)
                    idx += 1
                
        return snp_genotypes,gene_locus_map
                
    def simulate_embryo_sequence(self,snp_map):
        """
        Given seeded parental chromatids and a map of SNP locations for each transcript,
        simulate embryo sequences
        """
        updated_m0,updated_m1 = defaultdict(dict),defaultdict(dict)
        updated_p0,updated_p1 = defaultdict(dict),defaultdict(dict)
        for parent in ["mother","father"]:
            
            # Consider homologous recombination with respect to each chromosome
            for chr_num in np.sort(np.unique(self.input_df["chromosome"])):
                
                # Subset df for specific chr, sorted by start position 
                chr_df = self.input_df[self.input_df["chromosome"] == chr_num].sort_values(by=["start"])
                
                # Chiasma point chosen over range of transcripts reads
                transcript_range = range(chr_df["start"].iloc[0], chr_df["start"].iloc[-1]+len(chr_df["sequence"].iloc[-1])) 
                chiasma = random.choice(transcript_range)
                chiasma_flag = 1 # 1 means chiasma has not been reached; 0 means passed chiasma
                for idx,transcript_entry in chr_df.iterrows():
                    if parent == "mother":
                        chiasma_flag = self.homologous_recombination(
                            transcript_entry = transcript_entry,
                            updated_c0 = updated_m0,
                            updated_c1 = updated_m1,
                            c0 = self.m0_chromatid,
                            c1 = self.m1_chromatid,
                            chiasma = chiasma,
                            flag = chiasma_flag
                        )
                    else:
                        chiasma_flag = self.homologous_recombination(
                            transcript_entry = transcript_entry,
                            updated_c0 = updated_p0,
                            updated_c1 = updated_p1,
                            c0 = self.p0_chromatid,
                            c1 = self.p1_chromatid,
                            chiasma = chiasma,
                            flag = chiasma_flag
                        )
        m_chromatid = self.chromosome_inheritance(updated_m0, updated_m1)
        p_chromatid = self.chromosome_inheritance(updated_p0, updated_p1)
        
        return m_chromatid,p_chromatid
    
    def homologous_recombination(self,transcript_entry,updated_c0,updated_c1,c0,c1,chiasma,flag):
        """
        for a given transcript entry, simulate process of homologous recombination, given
        a uniformly selected chiasma point. 
        """    
        gene,chrom,start,end,direction,sequence = transcript_entry

        # 1 means chiasma has not been reached; 0 means passed chiasma
        if flag <= start and flag == 1:
            flag = 0
            
            # Case where chiasma point lies within a gene
            if chiasma in range(start,start+len(sequence)):
                updated_c0[chrom][gene] = c0[chrom][gene][:chiasma]+c1[chrom][gene][chiasma:]
                updated_c1[chrom][gene] = c1[chrom][gene][:chiasma]+c0[chrom][gene][chiasma:]
                return flag

        if flag == 1: # Before chiasma point has been reached
            updated_c0[chrom][gene] = c0[chrom][gene]
            updated_c1[chrom][gene] = c1[chrom][gene]
        else: # After chiasma point has been reached
            updated_c0[chrom][gene] = c1[chrom][gene]
            updated_c1[chrom][gene] = c0[chrom][gene]
        
        return flag
    
    def chromosome_inheritance(self,c0,c1):
        """
        Given homologous chromosomes, randomly select one to be inherited by embryo
        """
        chromatid = defaultdict(dict)
        for chrom in c0:
            chromatid[chrom] = c0[chrom] if random.random() <= 0.5 else c1[chrom]
        
        return chromatid

def adjustedHamming(g0, g1):
    if len(g0) != len(g1):
        raise ValueError("Sequence lengths unequal.")
    else:
        hd,adjlen = 0, 0
        for i, j in zip(g0, g1):
            if i == 0 or j == 0: 
                continue
            adjlen += 1
            if i != j: 
                hd += 1
        return hd, adjlen

def run(args):
    
    transcriptsPath = args.transcriptsPath
    countsPath = args.countsPath
    heterozygosity = args.heterozygosity
    n_embryos = args.n_embryos
    mean_cells = args.mean_cells
    sd_cells = args.sd_cells
    trial = args.trial
    outDir = args.outDir

    df = pd.read_csv(transcriptsPath)
    df = df.dropna() # 28240.t1 sequence is NaN

    count_df = pd.read_csv(countsPath)
    count_df = count_df[count_df["Lv_name"] != "LVA_m28240.t1"]

    simulator = EmbryoSequenceSimulator(df, n_embryos, mean_cells, sd_cells)
    X,true_clusters,snp_map = simulator.embryo_population_simulation(heterozygosity, count_df)

    num_snps = 0
    snp_map = simulator.global_seeding(heterozygosity)
    for chrom,genes in snp_map.items():
        for gene,snp_positions in genes.items():
            num_snps += len(snp_positions)

    n = X.shape[0]
    hd = np.zeros((n,n))
    adjlen = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            adjusted_hamming,length = adjustedHamming(X[i,:], X[j,:])
            hd[i,j] = adjusted_hamming
            adjlen[i,j] = length

    ddistX = np.divide(hd,adjlen)
    out_df = pd.DataFrame(ddistX)
    label = f"LVsimulation_{n_embryos}embryos_h{heterozygosity}_{num_snps}snps_mu{mean_cells}sd{sd_cells}_trial{trial}"
    out_df.to_csv(f"{outDir}/{label}_nhd.csv", index=False, header=False)

    with open(f"{outDir}/{label}_true_clusters.txt", "w") as f:
        for cluster_label in true_clusters:
            f.write(str(cluster_label)+"\n")

def main():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', '--transcriptsPath', type=str, required=True, help='')
    parser.add_argument('-c', '--countsPath', type=str, required=True, help='')
    parser.add_argument('-v', '--heterozygosity', type=float, required=True, help='')
    parser.add_argument('-n', '--n_embryos', type=int, required=True, help='')
    parser.add_argument('-m', '--mean_cells', type=int, required=True, help='')
    parser.add_argument('-s', '--sd_cells', type=int, required=True, help='')
    parser.add_argument('-i', '--trial', type=int, required=True, help='')
    parser.add_argument('-o', '--outDir', type=str, required=True, help='')
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()