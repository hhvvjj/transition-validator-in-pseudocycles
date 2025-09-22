# Transition Validator in Pseudocycles

[![Research](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15546925-orange.svg)](https://doi.org/10.5281/zenodo.15546925)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License](https://img.shields.io/badge/license-CC--BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

A Python tool for analysis and verification of pseudocycles highlighted in Collatz sequences represented using the tuple-based transform

## Mathematical Foundation

### Tuple-Based Transform

The reversible algorithm which transforms Collatz sequence values using the tuple $[p, f(p), m, q]$. A complete Tuple-based Transform calculator is available [here](https://github.com/hhvvjj/tuple-based-transform-calculator). The complete theoretical framework is detailed in this [article](http://dx.doi.org/10.5281/zenodo.15546925)

### Pseudocycle Detection

A search engine that systematically identifies $m_r$ values by analyzing $m$ repetitions and catalogs all discovered pairs. A complete pseudocycle detection tool is available [here](https://github.com/hhvvjj/tuple-transform-mr-pairs-finder). The complete theoretical framework is detailed in this [article](http://dx.doi.org/10.5281/zenodo.15546925)

### Valid $ω(m_r) = m_r$ Transitions

Implementation for rigorous pseudocycle verification in Collatz sequences through systematic validation of the $ω(m_r) = m_r$ condition. The tool provides step-by-step trace analysis of the transform operations. In the m-domain these operations become:

- $T1(m) = 3m + 1$ (when $p = 1$, corresponding to odd Collatz values)
- $T2(m) = ⌊m/2⌋$  (when $p = 2$, corresponding to even Collatz values)

The tool verifies whether applying the sequence of $T1$ or $T2$ transformations returns to the original $m_r$ value, validating the condition **$ω(m_r) = m_r$**.

## Computational Complexity

   - Time: $O(p)$ where $p$ is the pseudocycle length
   - Space: $O(p)$ for transformation steps storage

## Installation

### System Requirements

- Python 3.8 or higher
- No external dependencies required (uses only standard library)

### Package Installation

### Red Hat-based Systems (RHEL, CentOS, Fedora, Rocky Linux or AlmaLinux)

```bash
# RHEL/CentOS/Rocky/AlmaLinux 8+
sudo dnf install python3

# RHEL/CentOS 7
sudo yum install python3

# Fedora
sudo dnf install python3

# Verify installation
python3 --version
```
#### Debian-based Systems (Ubuntu, Debian or Linux Mint)

```bash
# Ubuntu/Debian/Mint
sudo apt update
sudo apt install python3 python3-pip

# Verify installation
python3 --version
```

### Setup

```bash
# Clone the repository
git clone https://github.com/hhvvjj/transition-validator-in-pseudocycles.git
cd transition-validator-in-pseudocycles
```

## Usage

```
python3 transition_validator_in_pseudocycles.py <n>
```

### Parameters:

- **n**: Initial positive integer for Collatz sequence (mandatory)

### Examples

```
# Basic Analysis
python3 transition_validator_in_pseudocycles.py 7
```

## Output

### Console Output

```
**************************************************************************
* Transition validator in pseudocycles                                   *
**************************************************************************

[*] PSEUDOCYCLES ENDS

	- Limits for Tuple-based Transform domain: mr = 3 (first at position 1, repeats at position 14, distance = 13)
	- Equivalent limits for Collatz domain: na = 7 (first occurrence) and nb = 8 (second occurrence)

[*] COMPREHENSIVE SEQUENCES ENUMERATION

	- Tuple-based Transform Details
		p-parameters sequence:
		[1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2]

		m-parameters sequence:
		[3, 10, 5, 16, 8, 25, 12, 6, 19, 9, 4, 2, 7, 3, 1, 0, 0, 1, 0]

	- Collatz sequence:
		[7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1, 4, 2, 1]

[*] PSEUDOCYCLE BETWEEN TWO mr = 3 VALUES

	- p-parameters subsequence:
	[1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2]

	- m-parameters subsequence:
	[3, 10, 5, 16, 8, 25, 12, 6, 19, 9, 4, 2, 7, 3]

[*] TRACE OF ω(mr) = mr WITHIN THE PSEUDOCYCLE

	 Step  Current m  p  Operation              Result   Status
	----- ---------- --- -------------------- -------- --------
	    1          3  -  Initial                     3  [Start]
	    2          3  1  T1: 3 · 3 + 1              10         
	    3         10  2  T2: ⌊10 ÷ 2⌋                5         
	    4          5  1  T1: 3 · 5 + 1              16         
	    5         16  2  T2: ⌊16 ÷ 2⌋                8         
	    6          8  1  T1: 3 · 8 + 1              25         
	    7         25  2  T2: ⌊25 ÷ 2⌋               12         
	    8         12  2  T2: ⌊12 ÷ 2⌋                6         
	    9          6  1  T1: 3 · 6 + 1              19         
	   10         19  2  T2: ⌊19 ÷ 2⌋                9         
	   11          9  2  T2: ⌊9 ÷ 2⌋                 4         
	   12          4  2  T2: ⌊4 ÷ 2⌋                 2         
	   13          2  1  T1: 3 · 2 + 1               7         
	   14          7  2  T2: ⌊7 ÷ 2⌋                 3     [OK]

[*] ANALYSIS OF TUPLE-BASED TRANSFORM PSEUDOCYCLE

	- Verification successful: pseudocycle from mr = 3 to mr = 3 validated
		ω(3) = 3
		T^13(3) = 3
		Transformation T1 was used 5 times
		Transformation T2 was used 8 times
		Total number of transformations was 13, equal to pseudocycle length, 13
```

## Contributing

Contributions are welcome! Please follow these guidelines:

**Code Contributions:**

- Maintain mathematical accuracy against the original article
- Preserve hash table integrity and parallel processing efficiency
- Follow existing documentation standards and code style

**Research Contributions:**

- Validate theoretical changes against sequence equivalence tests
- Provide mathematical proofs or references for algorithmic modifications
- Include performance benchmarks for optimization claims

## Citation

If you use this code in your research, please cite:

```bibtex
@software{transition_validator_in_pseudocycles,
  title={Transition Validator in Pseudocycles},
  author={Javier Hernandez},
  year={2025},
  url={https://github.com/hhvvjj/transition_validator_in_pseudocycles},
  note={Implementation based on research DOI: 10.5281/zenodo.15546925}
}
```

## Files

- `transition_validator_in_pseudocycles.py` - Main implementation
- `README.md` - This documentation
- `LICENSE` - License file

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

**You are free to:**

- Share — copy and redistribute the material
- Adapt — remix, transform and build upon the material

**Under the following terms:**

- Attribution — You must give appropriate credit
- NonCommercial — You may not use the material for commercial purposes
- ShareAlike — If you remix, transform or build upon the material, you must distribute your contributions under the same license

See [LICENSE](https://github.com/hhvvjj/transition-validator-in-pseudocycles/blob/main/LICENSE) for full details.

## Contact

For questions about the algorithm implementation, mathematical details or optimization strategies, please contact via email (271314@pm.me).
