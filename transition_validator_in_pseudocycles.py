# ***********************************************************************************
# TRANSITION VALIDATOR IN PSEUDOCYCLES - COLLATZ SEQUENCE ANALYSIS
# ***********************************************************************************
#
# Author: Javier Hernandez
#
# Email: 271314@pm.me
#
# Description:
# This tool implements pseudocycle analysis for Collatz sequences through the tuple-based 
# transform approach. The methodology transforms each Collatz value c into a corresponding
# m-value using m = (c - p)/2, where p encodes the parity (p=1 for odd, p=2 for even).
# This transformation enables systematic detection and verification of pseudocycles.
#
# Key Features:
# - Domain transformation: Maps Collatz sequences to tuple-based transform sequences
# - Pseudocycle detection: Identifies repeated mr values and their boundaries
# - Mathematical verification: Tests the condition ω(mr) = mr through T1/T2 operations
# - Cross-domain analysis: Maintains correspondence between Collatz and Tuple-based 
#   Transform domains
# - Complete traceability: Provides step-by-step verification of pseudocycle properties
#
# The transformation operations in the m-domain are:
#   T1(m) = 3m + 1  (when p = 1, corresponding to odd Collatz values)
#   T2(m) = ⌊m/2⌋   (when p = 2, corresponding to even Collatz values)
#
# Pseudocycle verification confirms whether applying the sequence of T1/T2 transformations 
# returns to the original mr value, validating ω(mr) = mr.
#
# This approach reveals structural properties of Collatz sequences through systematic analysis
# of repeated values in the transformed domain, potentially contributing to deeper understanding
# of the Collatz Conjecture.
#
# Usage:
# python3 transition_validator_in_pseudocycles.py <n>
#
# Example:
# python3 transition_validator_in_pseudocycles.py 27
#
# Output:
# - Comprehensive sequence enumeration (Collatz, m-parameters, p-parameters)
# - Pseudocycle boundary identification and cross-domain correspondence
# - Detailed verification trace of ω(mr) = mr condition
# - Statistical analysis of T1/T2 transformation frequencies
# - Mathematical conclusion with formal notation
#
# License:
# CC-BY-NC-SA 4.0 International
# For additional details, visit:
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# For full details, visit
# https://github.com/hhvvjj/transition_validator_in_pseudocycles/blob/main/LICENSE
#
# Research Reference:
# Based on the tuple-based transform methodology described in:
# https://doi.org/10.5281/zenodo.15546925
#
# ***********************************************************************************

# ***********************************************************************************
# * 1. STANDARD LIBRARY IMPORTS
# ***********************************************************************************


import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ***********************************************************************************
# * 2. CONFIGURATION AND DATA STRUCTURES
# ***********************************************************************************


@dataclass(frozen=True)
class SequenceData:
    collatz_seq: Tuple[int, ...]
    m_seq: Tuple[int, ...]
    p_seq: Tuple[int, ...]
    mr_value: int
    mr_first_pos: int
    mr_repeat_pos: int
    pseudocycle_length: int


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    t1_count: int
    t2_count: int
    transformation_steps: Tuple[Tuple[int, int, int], ...]  # (current, p_val, next_val)
    overflow_step: Optional[int] = None


@dataclass(frozen=True)
class Constants:
    MAX_SEQUENCE_LENGTH: int = 10000
    MAX_TRANSFORMATION_LENGTH: int = 100
    MAX_SAFE_VALUE: int = 2**64 - 1


# ***********************************************************************************
# * 3. CORE MATHEMATICAL OPERATIONS
# ***********************************************************************************


def calculate_p(c: int) -> int:
    """
    Calculate the parity parameter p for a given Collatz sequence value.
        
    This function determines the transformation type parameter used in the tuple-based 
    transform domain. The parameter p encodes whether the next Collatz operation will 
    be a division by 2 (even case) or a 3n+1 transformation (odd case).
        
    Mathematical definition:
        p(c) = { 2  if c ≡ 0 (mod 2)  [even case]
            { 1  if c ≡ 1 (mod 2)  [odd case]
        
    Args:
        c: A positive integer from the Collatz sequence
            
    Returns:
        int: The parity parameter (1 for odd input, 2 for even input)
            
    Example:
        >>> calculate_p(27)  # odd
        1
        >>> calculate_p(82)  # even  
        2

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return 2 if c % 2 == 0 else 1


def calculate_m(c: int) -> int:
    """
    Transform a Collatz domain value to its corresponding tuple-based transform domain value.
        
    This function implements the fundamental domain transformation m = (c - p)/2, which 
    maps Collatz sequence values to the tuple-based transform domain where pseudocycle 
    analysis becomes more tractable. This transformation preserves the essential 
    mathematical structure while enabling systematic detection of repeated values.
        
    Mathematical definition:
        m(c) = (c - p(c))/2 where p(c) is the parity parameter
        
    The inverse transformation is: c = 2m + p
        
    Args:
        c: A positive integer from the Collatz sequence
            
    Returns:
        int: The corresponding value in the tuple-based transform domain
            
    Example:
        >>> calculate_m(27)  # m = (27-1)/2 = 13
        13
        >>> calculate_m(82)  # m = (82-2)/2 = 40
        40

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return (c - calculate_p(c)) // 2


def collatz_step(n: int) -> int:
    """
    Apply a single iteration of the Collatz function.
        
    Implements the classical Collatz conjecture transformation rule, also known as 
    the 3n+1 function or Syracruse function. This is the fundamental operation that 
    generates the Collatz sequence.
        
    Mathematical definition:
        f(n) = { n/2      if n ≡ 0 (mod 2)  [even case]
               { 3n + 1   if n ≡ 1 (mod 2)  [odd case]
        
    Args:
        n: A positive integer
            
    Returns:
        int: The result of applying one Collatz transformation step
            
    Example:
        >>> collatz_step(27)  # odd: 3*27 + 1
        82
        >>> collatz_step(82)  # even: 82/2
        41

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return n // 2 if n % 2 == 0 else 3 * n + 1


def apply_transition(m: int, p: int) -> int:
    """
    Apply tuple-based transform domain transitions T1 or T2 based on parity parameter.
        
    This function implements the core transformations in the tuple-based transform domain,
    which are equivalent to Collatz operations but operate on m-values. The choice of
    transformation depends on the parity parameter p, maintaining mathematical equivalence
    with the original Collatz domain.
        
    Mathematical definitions:
        T1(m) = 3m + 1  when p = 1 (corresponds to odd Collatz values)
        T2(m) = ⌊m/2⌋   when p = 2 (corresponds to even Collatz values)
        
    Domain equivalence:
        If c_i → c_{i+1} in Collatz domain, then m_i → m_{i+1} via T1 or T2
        where m_i = (c_i - p_i)/2 and m_{i+1} = (c_{i+1} - p_{i+1})/2
        
    Args:
        m: A value in the tuple-based transform domain
        p: Parity parameter (1 or 2) determining the transformation type
            
    Returns:
        int: The result of applying transformation T1 or T2 to m
            
    Example:
        >>> apply_transition(13, 1)  # T1: 3 · 13 + 1
        40
        >>> apply_transition(40, 2)  # T2: ⌊40 ÷ 2⌋
        20

    Time Complexity: O(1)
    Time Space: O(1)
    """
    return 3 * m + 1 if p == 1 else m // 2


# ***********************************************************************************
# * 4. SEQUENCE GENERATION
# ***********************************************************************************


def generate_collatz_sequence(n: int, constants: Constants) -> Tuple[int, ...]:
    """
    Generate a complete Collatz sequence with controlled termination and trivial cycle extension.
        
    This function generates the Collatz sequence starting from n, continuing until reaching 1
    or hitting computational limits. Upon reaching 1, it extends the sequence with one complete
    trivial cycle (1 → 4 → 2 → 1) to ensure sufficient data for pseudocycle analysis in cases
    where the repetition occurs within the trivial cycle, mr=0.
        
    The function includes overflow protection to prevent infinite computation when dealing
    with large odd numbers that could exceed safe integer limits during the 3n+1 operation.
        
    Termination conditions:
        1. Reaches value 1 (normal termination + trivial cycle extension)
        2. Exceeds MAX_SEQUENCE_LENGTH (computational limit)
        3. Risk of integer overflow detected (safety limit)
        
    Args:
        n: Initial positive integer to start the Collatz sequence
        constants: Configuration object containing computational limits
            
    Returns:
        Tuple[int, ...]: Immutable sequence of Collatz values, potentially including
                        the trivial cycle extension [1, 4, 2, 1]
            
    Example:
        >>> generate_collatz_sequence(5, constants)
        (5, 16, 8, 4, 2, 1, 4, 2, 1)

    Time Complexity: O(k) where k is the sequence length (bounded by MAX_SEQUENCE_LENGTH)
    Space Complexity: O(k) for storing the sequence

    Note:
        The trivial cycle extension ensures that sequences terminating at 1 have
        sufficient repetitive structure for meaningful pseudocycle analysis.
    """
    sequence = [n]
    current = n
    
    # Generate until we reach 1
    while len(sequence) < constants.MAX_SEQUENCE_LENGTH and current != 1:
        if current > constants.MAX_SAFE_VALUE // 3 and current % 2 == 1:
            break
        current = collatz_step(current)
        sequence.append(current)
    
    # If we reached 1, add one cycle of trivial loop: 1 → 4 → 2 → 1
    if current == 1 and len(sequence) > 1:
        sequence.extend([4, 2, 1])
    
    return tuple(sequence)


def derive_m_p_sequences(collatz_seq: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Derive the corresponding m-sequence and p-sequence from a Collatz sequence.
        
    This function performs the fundamental domain transformation by computing the tuple-based
    transform domain sequence (m-sequence) and its associated parity parameter sequence 
    (p-sequence) from the input Collatz sequence. The transformation establishes a bijective
    correspondence between the Collatz domain and the tuple-based transform domain.
        
    Mathematical transformation:
        For each c_i in collatz_seq[:-1]:  (excluding the last element)
            m_i = (c_i - p_i) / 2
            p_i = 2 if c_i is even, 1 if c_i is odd
        
    Sequence length relationship:
        len(m_seq) = len(p_seq) = len(collatz_seq) - 1
        
    The last element of the Collatz sequence is excluded because it represents the target
    of the final transformation, not a source for further transformation.
        
    Args:
        collatz_seq: Immutable sequence of Collatz values
            
    Returns:
        Tuple containing:
            - m_seq: Tuple of values in the tuple-based transform domain
            - p_seq: Tuple of parity parameters (1 or 2) for each transformation
            
    Example:
        >>> derive_m_p_sequences((27, 82, 41, 124))
        ((13, 40, 20), (1, 2, 1))

    Time Complexity: O(n) where n = len(collatz_seq)
    Space Complexity: O(n) for the two output sequences
               
    Note:
        Returns empty tuples if input sequence has fewer than 2 elements,
        as meaningful transformation requires at least one transition.
    """
    if not collatz_seq or len(collatz_seq) < 2:
        return (), ()
    
    # Generate m and p only for elements up to the penultimate (exclude the last)
    # This ensures m and p sequences have length = len(collatz_seq) - 1
    values_to_process = collatz_seq[:-1]  # All except the last element
    
    m_seq = tuple(calculate_m(c) for c in values_to_process)
    p_seq = tuple(calculate_p(c) for c in values_to_process)
    
    return m_seq, p_seq


def generate_all_sequences(n: int, constants: Constants) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """
    Generate the complete triad of sequences: Collatz, tuple-based transform, and parity parameters.
        
    This function orchestrates the complete sequence generation pipeline, producing all three
    fundamental sequences required for pseudocycle analysis. It combines Collatz sequence
    generation with domain transformation to provide a comprehensive mathematical representation
    of the dynamical system in both domains.
        
    Pipeline operations:
        1. Generate Collatz sequence from initial value n
        2. Transform to tuple-based domain (m-sequence)
        3. Extract parity parameter sequence (p-sequence)
        
    Mathematical relationship:
        collatz_seq[i] ↔ (m_seq[i], p_seq[i]) via bijective transformation
        where collatz_seq[i] = 2 * m_seq[i] + p_seq[i]
        
    Args:
        n: Initial positive integer for sequence generation
        constants: Configuration object containing computational limits and safety parameters
            
    Returns:
        Tuple containing three immutable sequences:
            - collatz_seq: Original Collatz domain sequence
            - m_seq: Tuple-based transform domain sequence  
            - p_seq: Parity parameter sequence encoding transformation types
            
    Example:
        >>> generate_all_sequences(5, constants)
        ((5, 16, 8, 4, 2, 1, 4, 2, 1), (2, 7, 3, 1, 0, 2, 1, 0), (1, 2, 2, 2, 2, 1, 2, 2))

    Time Complexity: O(k) where k is the Collatz sequence length
    Space Complexity: O(k) for storing all three sequences

    Note:
        The three sequences maintain strict mathematical correspondence and identical
        indexing for consistent cross-domain analysis.
    """
    collatz_seq = generate_collatz_sequence(n, constants)
    m_seq, p_seq = derive_m_p_sequences(collatz_seq)
    return collatz_seq, m_seq, p_seq


# ***********************************************************************************
# * 5. PATTERN ANALYSIS
# ***********************************************************************************


def find_first_repetition(sequence: Tuple[int, ...]) -> Tuple[Optional[int], int, int]:
    """
    Identify the first repeated value and its occurrence positions in a sequence.
        
    This function performs sequential scan analysis to detect the first value that appears
    more than once in the sequence, which is fundamental for pseudocycle identification
    in the tuple-based transform domain. The detection of repeated m-values (mr) establishes
    the boundaries of pseudocycles where the condition ω(mr) = mr can be verified.
        
    Algorithm:
        Single-pass sequential scan using hash table for O(1) lookup complexity.
        Returns immediately upon detecting the first repetition to ensure minimal
        computational overhead and deterministic behavior.
        
    Mathematical significance:
        The first repeated value mr defines the pseudocycle boundaries:
        - First occurrence: start of potential pseudocycle
        - Second occurrence: end of pseudocycle, establishing cycle length
        
    Args:
        sequence: Immutable sequence of integers (typically m-sequence)
            
    Returns:
        Tuple containing:
            - repeated_value: The first value that appears twice (None if no repetition)
            - first_position: Index of first occurrence of repeated value (0-based, internal use)
            - second_position: Index of second occurrence of repeated value (0-based, internal use)
            
    Example:
        >>> find_first_repetition((1, 3, 7, 2, 5, 3, 8))
        (3, 1, 5)  # Value 3 first appears at index 1, repeats at index 5 (0-based internal)

    Time Complexity: O(n) where n = len(sequence), worst case when repetition is mr=0
    Space Complexity: O(n) for the hash table in worst case

    Note:
        Returns (None, -1, -1) if no repetition is found within the sequence.
        For pseudocycle analysis, the absence of repetition indicates either
        divergent behavior or computational limits being reached.
    """
    seen = {}
    
    for i, value in enumerate(sequence):
        if value in seen:
            return value, seen[value], i
        seen[value] = i
    
    return None, -1, -1


def create_sequence_data(collatz_seq: Tuple[int, ...], m_seq: Tuple[int, ...], p_seq: Tuple[int, ...]) -> Optional[SequenceData]:
    """
    Construct a comprehensive SequenceData object encapsulating all pseudocycle analysis parameters.
        
    This function performs the critical integration step that combines the three fundamental
    sequences into a unified data structure, identifies the first repeated value (mr) in the
    m-sequence, and computes all derived pseudocycle parameters necessary for mathematical
    verification of the condition ω(mr) = mr.

    Pseudocycle parameter computation:
        - mr_value: First repeated value in m-sequence (defines pseudocycle identity)
        - mr_first_pos: Index of first occurrence (pseudocycle start boundary, 0-based internal)
        - mr_repeat_pos: Index of repetition (pseudocycle end boundary, 0-based internal)
        - pseudocycle_length: Distance between occurrences (number of transformations)
        
    Note: Positions are stored internally as 0-based indices but displayed as 1-based for users.
        
    Mathematical significance:
        The SequenceData object encapsulates the complete mathematical state required
        to verify pseudocycle properties and establish correspondence between Collatz
        and tuple-based transform domains within the identified cycle boundaries.
        
    Args:
        collatz_seq: Complete Collatz domain sequence
        m_seq: Corresponding tuple-based transform domain sequence
        p_seq: Parity parameter sequence encoding transformation types
            
    Returns:
        SequenceData: Immutable object containing all sequences and pseudocycle parameters,
                    or None if no repetition is found in m_seq or sequences are empty
            
    Example:
        If m_seq = (13, 40, 20, 61, 30, 15, 46, 23, 70, 35, 106, 53, 160, 80, 40, ...):
        Returns SequenceData with mr_value=40, mr_first_pos=1, mr_repeat_pos=14, 
        pseudocycle_length=13

    Time Complexity: O(n) where n = len(m_seq) due to find_first_repetition call
    Space Complexity: O(n) for storing sequences in SequenceData object

    Note:
        Returns None for sequences without repetitions, indicating either divergent
        dynamics or computational limits preventing pseudocycle detection.
    """
    if not m_seq:
        return None
    
    mr_value, mr_first_pos, mr_repeat_pos = find_first_repetition(m_seq)
    
    if mr_value is None:
        return None
    
    pseudocycle_length = mr_repeat_pos - mr_first_pos
    
    return SequenceData(
        collatz_seq=collatz_seq,
        m_seq=m_seq,
        p_seq=p_seq,
        mr_value=mr_value,
        mr_first_pos=mr_first_pos,
        mr_repeat_pos=mr_repeat_pos,
        pseudocycle_length=pseudocycle_length
    )


# ***********************************************************************************
# * 6. PSEUDOCYCLE OPERATIONS
# ***********************************************************************************


def calculate_pseudocycle_endpoints(data: SequenceData) -> Tuple[int, int]:
    """
    Compute the corresponding Collatz domain endpoints for a detected pseudocycle.
        
    This function establishes the bijective correspondence between pseudocycle boundaries
    in the tuple-based transform domain (defined by repeated mr values) and their
    equivalent positions in the original Collatz domain. The endpoints na and nb represent
    the Collatz values that correspond to the same mr value at different positions.
        
    Mathematical transformation:
        Given mr at positions (first_pos, repeat_pos) with parity parameters (p_first, p_repeat):
        na = 2 * mr + p_first   (Collatz value at first mr occurrence)
        nb = 2 * mr + p_repeat  (Collatz value at repeated mr occurrence)
        
    Domain correspondence significance:
        The pseudocycle in tuple-based domain: mr → ... → mr
        Corresponds to Collatz domain segment: na → ... → nb
        where both endpoints map to the same m-value but may have different Collatz values
        due to different parity parameters.
        
    Args:
        data: SequenceData object containing detected pseudocycle information
            
    Returns:
        Tuple containing:
            - na: Collatz domain value corresponding to first mr occurrence
            - nb: Collatz domain value corresponding to repeated mr occurrence
            
    Example:
        If mr=40 at positions (1,14) with p_values (2,2):
        na = 2 * 40 + 2 = 82, nb = 2 * 40 + 2 = 82
        
        If mr=15 at positions (5,12) with p_values (1,2):
        na = 2 * 15 + 1 = 31, nb = 2 * 15 + 2 = 32

    Time Complexity: O(1)
    Space Complexity: O(1)
        
    Note:
        When na = nb, the pseudocycle represents identical Collatz values.
        When na ≠ nb, it represents different Collatz values mapping to the same mr.
    """
    p_first = data.p_seq[data.mr_first_pos]
    p_repeat = data.p_seq[data.mr_repeat_pos]
    na = 2 * data.mr_value + p_first
    nb = 2 * data.mr_value + p_repeat
    return na, nb


def get_sequence_slice(sequence: Tuple[int, ...], start: int, end: int) -> Tuple[int, ...]:
    """
    Extract a contiguous subsequence between specified inclusive boundaries.
        
    This utility function performs controlled subsequence extraction with inclusive
    endpoint semantics, ensuring consistent behavior across all pseudocycle analysis
    operations. The inclusive end parameter differs from Python's standard slicing
    convention to maintain mathematical clarity when defining cycle boundaries.
        
    Boundary semantics:
        Extracts elements from index 'start' through index 'end' (both inclusive)
        Equivalent to sequence[start:end+1] in standard Python slicing notation
        
    Mathematical application:
        Used to extract pseudocycle segments where both boundary positions correspond
        to meaningful mathematical events (e.g., repeated mr values), requiring
        inclusion of both endpoints in the analysis.
        
    Args:
        sequence: Source immutable sequence for extraction
        start: Starting index (inclusive) for subsequence extraction
        end: Ending index (inclusive) for subsequence extraction
            
    Returns:
        Tuple[int, ...]: Immutable subsequence containing elements from start to end
        
    Example:
        >>> get_sequence_slice((10, 20, 30, 40, 50), 1, 3)
        (20, 30, 40)  # Indices 1, 2, 3 (inclusive)

    Time Complexity: O(k) where k = end - start + 1 (length of extracted slice)
    Space Complexity: O(k) for the new tuple

    Note:
        This function assumes valid indices within sequence bounds.
        Used consistently across all pseudocycle extraction operations to maintain
        uniform boundary semantics throughout the analysis pipeline.
    """
    return sequence[start:end + 1]


def get_m_pseudocycle(data: SequenceData) -> Tuple[int, ...]:
    """
    Extract the tuple-based transform domain pseudocycle segment between repeated mr values.
        
    This function isolates the core pseudocycle in the tuple-based transform domain,
    spanning from the first occurrence of the repeated value mr to its reappearance.
    This segment represents the fundamental mathematical object where the condition
    ω(mr) = mr is verified through sequential application of T1 and T2 transformations.
        
    Pseudocycle structure:
        Extracts m_seq[mr_first_pos : mr_repeat_pos + 1] (inclusive endpoints)
        Format: (mr, intermediate_values..., mr)
        The endpoints are identical by definition (both equal to mr_value)
        
    Mathematical significance:
        This is the primary object of analysis for pseudocycle verification.
        The sequence represents a closed orbit in the tuple-based transform domain
        where applying the transformation chain T^n returns to the starting value:
        T^n(mr) = mr, where n = pseudocycle_length
        
    Args:
        data: SequenceData object containing detected pseudocycle parameters
            
    Returns:
        Tuple[int, ...]: Tuple-based transform domain subsequence representing
                        the complete pseudocycle from mr to mr
        
    Example:
        If m_seq = (13, 40, 20, 61, 30, 15, 46, 23, 70, 35, 106, 53, 160, 80, 40, ...)
        and mr=40 at positions (1, 14), then returns:
        (40, 20, 61, 30, 15, 46, 23, 70, 35, 106, 53, 160, 80, 40)

    Time Complexity: O(k) where k = pseudocycle_length
    Space Complexity: O(k) for the extracted subsequence        

    Note:
        The first and last elements are always identical (both equal mr_value).
        The intermediate elements represent the trajectory of the tuple-based
        transform dynamic system within the detected cycle.
    """
    return get_sequence_slice(data.m_seq, data.mr_first_pos, data.mr_repeat_pos)


def get_p_pseudocycle(data: SequenceData) -> Tuple[int, ...]:
    """
    Extract the parity parameter sequence corresponding to the tuple-based transform pseudocycle.
        
    This function isolates the sequence of transformation indicators (p-values) that
    define which transformations (T1 or T2) are applied during the pseudocycle evolution.
    The p-sequence provides the operational blueprint for verifying the condition ω(mr) = mr
    by specifying the exact sequence of T1 and T2 operations required to return to mr.
        
    Transformation encoding:
        p = 1 → Apply T1(m) = 3m + 1 (corresponds to odd Collatz values)
        p = 2 → Apply T2(m) = ⌊m/2⌋ (corresponds to even Collatz values)
        
    Verification role:
        The extracted p-sequence defines the transformation chain:
        mr → T_{p1} → T_{p2} → ... → T_{pn} → mr
        where each T_{pi} is determined by the corresponding p-value
        
    Args:
        data: SequenceData object containing pseudocycle boundary information

    Returns:
        Tuple[int, ...]: Sequence of parity parameters (1s and 2s) defining
                        the transformation operations within the pseudocycle

    Example:
        If p_seq = (1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, ...)
        and pseudocycle spans positions (1, 14), then returns:
        (2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2)

    Time Complexity: O(k) where k = pseudocycle_length
    Space Complexity: O(k) for the extracted subsequence    

    Note:
        The length of p_pseudocycle equals the pseudocycle_length, representing
        the exact number of T1/T2 transformations needed to complete one cycle.
        This sequence is essential for step-by-step verification of ω(mr) = mr.
    """
    return get_sequence_slice(data.p_seq, data.mr_first_pos, data.mr_repeat_pos)


# ***********************************************************************************
# * 7. VERIFICATION ENGINE
# ***********************************************************************************


def verify_pseudocycle_transitions(data: SequenceData, constants: Constants) -> VerificationResult:
    """
    Perform rigorous mathematical verification of the pseudocycle condition ω(mr) = mr.
        
    This function constitutes the core verification engine that tests whether the detected
    pseudocycle satisfies the fundamental mathematical property ω(mr) = mr through sequential
    application of tuple-based transform operations T1 and T2. The verification process
    traces the complete transformation chain starting from mr and validates return to the
    same value after exactly pseudocycle_length steps.
        
    Verification algorithm:
        1. Initialize current = mr_value
        2. For each p_i in the pseudocycle p-sequence:
            - Apply T1(current) = 3*current + 1 if p_i = 1
            - Apply T2(current) = ⌊current/2⌋ if p_i = 2
            - Record transformation step (current, p_i, result)
        3. Verify final result equals mr_value
        
    Mathematical significance:
        Success validates that the pseudocycle represents a true closed orbit in the
        tuple-based transform domain, confirming the conjecture that repeated mr values
        correspond to mathematically consistent cycle structures.
        
    Overflow protection:
        Monitors intermediate values to prevent integer overflow during T1 operations,
        ensuring computational safety when dealing with large mr values.
        
    Args:
        data: SequenceData containing pseudocycle parameters and p-sequence
        constants: Configuration object with computational safety limits
            
    Returns:
        VerificationResult containing:
            - success: Boolean indicating whether ω(mr) = mr holds
            - t1_count: Number of T1 transformations applied
            - t2_count: Number of T2 transformations applied  
            - transformation_steps: Complete trace of verification process
            - overflow_step: Step number where overflow occurred (if applicable)
        
    Example:
        For mr=40 with p_sequence=(2,1,2,2,1):
        40 →[T2]→ 20 →[T1]→ 61 →[T2]→ 30 →[T2]→ 15 →[T1]→ 46 → ... → 40
        Returns VerificationResult(success=True, t1_count=2, t2_count=3, ...)

    Time Complexity: O(k) where k = pseudocycle_length
    Space Complexity: O(k) for storing transformation steps

        Note:
        A successful verification provides strong evidence for the mathematical
        consistency of the tuple-based transform approach to Collatz analysis.
    """
    if data.pseudocycle_length == 0:
        return VerificationResult(
            success=True,
            t1_count=0,
            t2_count=0,
            transformation_steps=()
        )
    
    p_values = data.p_seq[data.mr_first_pos:data.mr_first_pos + data.pseudocycle_length]
    current = data.mr_value
    transformation_steps = []
    
    for i, p_val in enumerate(p_values):
        if current > constants.MAX_SAFE_VALUE // 3:
            return VerificationResult(
                success=False,
                t1_count=0,
                t2_count=0,
                transformation_steps=tuple(transformation_steps),
                overflow_step=i
            )
        
        next_val = apply_transition(current, p_val)
        transformation_steps.append((current, p_val, next_val))
        current = next_val
    
    success = current == data.mr_value
    t1_count = sum(1 for p in p_values if p == 1)
    t2_count = sum(1 for p in p_values if p == 2)
    
    return VerificationResult(
        success=success,
        t1_count=t1_count,
        t2_count=t2_count,
        transformation_steps=tuple(transformation_steps)
    )


# ***********************************************************************************
# * 8. OUTPUT FORMATTING
# ***********************************************************************************


def format_highlighted_sequence(sequence: Tuple[int, ...], highlight_positions: Tuple[int, ...]) -> str:
    """
    Generate formatted string representation of a sequence with visual emphasis on key positions.
        
    This function creates a human-readable representation of numerical sequences with
    selective highlighting of mathematically significant positions using ANSI color codes.
    The highlighting serves to draw attention to critical values such as repeated mr elements,
    pseudocycle boundaries, or transformation endpoints in the analysis output.
        
    Visual formatting:
        - Standard elements: Plain text representation
        - Highlighted elements: Green background with ANSI escape sequences
        - Format: "[elem1, elem2, HIGHLIGHTED_ELEM3, elem4, ...]"
        
    Mathematical application:
        Used extensively throughout the analysis pipeline to emphasize:
        - Repeated mr values in m-sequences
        - Pseudocycle boundary positions
        - Corresponding Collatz domain endpoints
        - Critical p-parameter positions for verification
        
    Args:
        sequence: Immutable sequence of integers to be formatted
        highlight_positions: Tuple of indices indicating which elements to emphasize
            
    Returns:
        str: Formatted string with bracketed sequence notation and color highlighting
        
    Example:
        >>> format_highlighted_sequence((10, 20, 30, 40, 50), (1, 3))
        "[10, \033[42m20\033[0m, 30, \033[42m40\033[0m, 50]"
        
    Time Complexity: O(n + h) where n = len(sequence), h = len(highlight_positions)
    Space Complexity: O(n) for the result string

    Note:
        The function applies highlighting non-destructively, preserving original
        sequence data while enhancing visual presentation for analysis interpretation.
        ANSI codes ensure compatibility with terminal environments supporting color output.
    """
    GREEN_BG = '\033[42m'
    RESET = '\033[0m'
    
    result = []
    for i, value in enumerate(sequence):
        if i in highlight_positions:
            result.append(f"{GREEN_BG}{value}{RESET}")
        else:
            result.append(str(value))
    
    return "[" + ", ".join(result) + "]"


def format_verification_table_header() -> str:
    """
    Generate standardized table header for pseudocycle verification trace display.
        
    This function creates a formatted header that establishes the tabular structure
    for presenting the step-by-step verification of the condition ω(mr) = mr. The header
    provides clear column organization for tracking transformation steps, intermediate
    values, operation types, and verification status throughout the pseudocycle analysis.
        
    Table structure specification:
        - Step: Sequential numbering of transformation operations
        - Current m: The m-value before applying transformation
        - p: Parity parameter (1 or 2) determining transformation type
        - Operation: Mathematical description of T1 or T2 transformation
        - Result: The m-value after applying transformation
        - Status: Verification indicators ([Start], [OK], [ERROR])
        
    Formatting standards:
        - Fixed-width columns for consistent alignment
        - Right-aligned numerical fields for decimal point alignment
        - Center-aligned single-character fields for visual balance
        - Separator line using dashes for clear header demarcation
        
    Mathematical context:
        The tabular format facilitates systematic verification by providing
        clear visibility into each transformation step of the ω(mr) = mr
        validation process, enabling identification of specific failure points
        or confirmation of successful pseudocycle completion.
        
    Returns:
        str: Multi-line formatted header string with column titles and separator
        
    Example output:
            Step  Current m   p Operation           Result   Status
            ----- ---------- --- -------------------- -------- --------
        
    Note:
        Designed for monospace terminal output to ensure proper column alignment
        across different sequence lengths and numerical ranges.
    """
    header = f"\t{'Step':>5} {'Current m':>10} {'p':^3} {'Operation':<20} {'Result':>8} {'Status':>8}\n"
    header += f"\t{'-'*5} {'-'*10} {'-'*3} {'-'*20} {'-'*8} {'-'*8}"
    return header


def format_verification_step(step: int, current: int, p_val: int, next_val: int, 
                           is_initial: bool = False, is_final: bool = False, mr_value: int = 0) -> str:
    """
    Format a single transformation step in the pseudocycle verification trace table.
        
    This function generates a standardized table row representing one step in the
    ω(mr) = mr verification process. It handles three distinct formatting cases:
    initial state presentation, intermediate transformation steps, and final
    verification status, providing complete traceability of the mathematical
    validation process.
        
    Row formatting variants:
        - Initial row: Displays starting mr value with "[Start]" status indicator
        - Transformation row: Shows T1/T2 operation with mathematical notation
        - Final row: Includes verification status ([OK] or [ERROR]) based on success
        
    Mathematical notation:
        - T1 operations: "T1: 3 · current + 1" for parity parameter p=1
        - T2 operations: "T2: ⌊current ÷ 2⌋" for parity parameter p=2
        - Unicode symbols (·, ⌊, ⌋) provide precise mathematical representation
        
    Status indicators:
        - "[Start]": Initial mr value before transformations
        - "[OK]": Final step where result equals mr_value (successful verification)
        - "[ERROR]": Final step where result ≠ mr_value (failed verification)
        - Empty: Intermediate steps without terminal status
        
    Args:
        step: Sequential step number in verification process (1-based for display)
        current: Current m-value before transformation
        p_val: Parity parameter (1 or 2) determining transformation type
        next_val: Resulting m-value after transformation
        is_initial: Boolean indicating if this is the starting row
        is_final: Boolean indicating if this is the concluding row
        mr_value: Original mr value for comparison in final verification
            
    Returns:
        str: Formatted table row with proper column alignment and mathematical notation
        
    Example:
        >>> format_verification_step(1, 40, 2, 20, mr_value=40)
        "    1         40   2 T2: ⌊40 ÷ 2⌋            20        "
        
    Note:
        Column widths and alignment match the header format exactly to ensure
        consistent tabular presentation across the complete verification trace.
    """
    if is_initial:
        return f"\t{1:>5} {mr_value:>10} {'-':^3} {'Initial':<20} {mr_value:>8} {'[Start]':>8}"
    
    operation = f"T1: 3 · {current} + 1" if p_val == 1 else f"T2: ⌊{current} ÷ 2⌋"
    status = "[OK]" if is_final and next_val == mr_value else ("[ERROR]" if is_final else "")
    
    return f"\t{step:>5} {current:>10} {p_val:^3} {operation:<20} {next_val:>8} {status:>8}"


# ***********************************************************************************
# * 9. DISPLAY OPERATIONS
# ***********************************************************************************

def display_program_header() -> None:
    """
    Output the standardized program identification header to the console.
        
    This function generates a formatted banner that identifies the program and its
    scientific purpose. The header serves as the primary visual identifier for the
    transition validator in pseudocycles, establishing the analytical context for
    subsequent mathematical output and verification results.
        
    Header specifications:
        - Fixed-width banner with asterisk border formatting
        - Centered program title with descriptive subtitle
        - Consistent spacing for professional presentation
        - Terminal-width optimization for standard console displays
        
    Scientific context:
        The header immediately establishes that this is a specialized mathematical
        analysis tool focused on Collatz sequence pseudocycle validation, preparing
        users for the technical nature of subsequent output and ensuring proper
        interpretation of verification results.
        
    Side effects:
        - Writes formatted header text to stdout
        - Modifies terminal display state
        - No return value (pure side effect function)
        
    Output format:
        **************************************************************************
        * Transition validator in pseudocycles                                   *
        **************************************************************************
        
    Note:
        This function is called once at program initialization to establish
        visual context before any mathematical analysis begins. The formatting
        is designed for clarity in both interactive terminal sessions and
        logged output capture scenarios.
    """
    print("")
    print("*" * 74)
    print("* Transition validator in pseudocycles                                   *")
    print("*" * 74)
    print("")


def display_pseudocycle_boundaries(data: SequenceData) -> None:
    """
    Output comprehensive information about detected pseudocycle boundary parameters.
        
    This function presents the fundamental mathematical parameters that define the
    detected pseudocycle in both the tuple-based transform domain and the corresponding
    Collatz domain. The output establishes the mathematical foundation for subsequent
    verification by clearly delineating the cycle boundaries and their cross-domain
    correspondence.
        
    Boundary parameter display:
        - mr value: The repeated value in the tuple-based transform domain
        - Position information: First occurrence and repetition indices (displayed as 1-based)
        - Cycle metrics: Distance between occurrences (pseudocycle length)
        - Domain correspondence: Equivalent na and nb values in Collatz domain
        
    Mathematical significance presentation:
        The function emphasizes the bijective relationship between domains by
        displaying both the abstract tuple-based parameters (mr, positions, distance)
        and their concrete Collatz domain manifestations (na, nb values), enabling
        users to understand the cross-domain mathematical structure.
        
    Cross-domain calculation:
        Computes and displays na = 2*mr + p_first and nb = 2*mr + p_repeat,
        demonstrating the explicit transformation that maps tuple-based boundaries
        to their Collatz domain equivalents.
        
    Side effects:
        - Writes pseudocycle boundary analysis to stdout
        - Displays mathematical parameters in structured format
        - Establishes context for subsequent sequence enumeration
        
    Output sections:
        [*] PSEUDOCYCLES ENDS
            - Tuple-based Transform domain limits
            - Equivalent Collatz domain limits
        
    Note:
        This display function provides essential mathematical context that enables
        interpretation of subsequent sequence displays and verification results.
        The information is critical for understanding the scope and nature of the
        detected pseudocycle structure.
    """
    print("[*] PSEUDOCYCLES ENDS")
    print("")
    
    print(f"\t- Limits for Tuple-based Transform domain: mr = {data.mr_value} (first at position {data.mr_first_pos + 1}, "
          f"repeats at position {data.mr_repeat_pos + 1}, distance = {data.pseudocycle_length})")
    
    na, nb = calculate_pseudocycle_endpoints(data)
    print(f"\t- Equivalent limits for Collatz domain: na = {na} (first occurrence) and nb = {nb} (second occurrence)")
    print("")


def display_all_sequences(data: SequenceData) -> None:
    """
    Present comprehensive enumeration of all three fundamental sequences with strategic highlighting.
        
    This function provides complete visual representation of the mathematical transformation
    pipeline, displaying the p-parameter sequence, m-parameter sequence, and Collatz sequence
    with selective highlighting of pseudocycle boundaries and mathematically significant
    positions. The presentation enables detailed cross-domain analysis and verification
    of sequence correspondence.
        
    Sequence presentation order and rationale:
        1. p-parameters: Transformation indicators showing T1/T2 operation types
        2. m-parameters: Tuple-based transform domain with highlighted mr repetitions
        3. Collatz sequence: Original domain with corresponding highlighted positions
        
    Strategic highlighting logic:
        - p-sequence: Highlights first and penultimate positions (verification range)
        - m-sequence: Highlights mr_first_pos and mr_repeat_pos (cycle boundaries)
        - Collatz sequence: Context-sensitive highlighting based on mr characteristics
        
    Special case handling:
        For consecutive zero repetitions (mr=0 in trivial cycle), Collatz highlighting
        targets destination values rather than source positions, providing more
        meaningful visual emphasis for the specific mathematical structure.
        
    Mathematical context visualization:
        The coordinated highlighting across all three sequences enables users to
        trace the mathematical relationships between domains and understand how
        pseudocycle boundaries manifest in each representational framework.
        
    Side effects:
        - Writes complete sequence enumeration to stdout
        - Applies color highlighting for mathematical emphasis
        - Displays tuple-based transform details and Collatz correspondence
        
    Output sections:
        [*] COMPREHENSIVE SEQUENCES ENUMERATION
            - Tuple-based Transform Details
                - p-parameters sequence (with verification range highlighting)
                - m-parameters sequence (with mr boundary highlighting)
            - Collatz sequence (with corresponding position highlighting)
        
    Note:
        This comprehensive display is essential for understanding the complete
        mathematical state and provides the visual foundation for interpreting
        subsequent pseudocycle analysis and verification results.
    """
    print("[*] COMPREHENSIVE SEQUENCES ENUMERATION")
    print("")
    print("\t- Tuple-based Transform Details")
    print("\t\tp-parameters sequence:")
    
    # For p sequence: highlight first and penultimate positions (verification range)
    if data.pseudocycle_length > 0 and data.mr_first_pos < len(data.p_seq) - 1:
        p_highlight_positions = (data.mr_first_pos, data.mr_first_pos + data.pseudocycle_length - 1)
    else:
        p_highlight_positions = ()
    
    # p sequence
    formatted_p = format_highlighted_sequence(data.p_seq, p_highlight_positions)
    print(f"\t\t{formatted_p}")
    print("")
    
    # m sequence - highlight mr positions
    m_highlight_positions = (data.mr_first_pos, data.mr_repeat_pos)
    print("\t\tm-parameters sequence:")
    formatted_m = format_highlighted_sequence(data.m_seq, m_highlight_positions)
    print(f"\t\t{formatted_m}")
    print("")
    
    # Collatz sequence - special case for mr=0 (consecutive zeros in trivial cycle)
    if data.mr_value == 0 and data.mr_repeat_pos == data.mr_first_pos + 1:
        # For consecutive zeros, highlight the destination values of transitions
        collatz_highlight_positions = (data.mr_first_pos + 1, data.mr_repeat_pos + 1)
    else:
        # For normal cases, use same positions as m sequence
        collatz_highlight_positions = m_highlight_positions
    
    print("\t- Collatz sequence:")
    formatted_collatz = format_highlighted_sequence(data.collatz_seq, collatz_highlight_positions)
    print(f"\t\t{formatted_collatz}")
    print("")


def display_pseudocycle_sequences(data: SequenceData) -> None:
    """
    Present isolated pseudocycle subsequences with focused highlighting of verification elements.
        
    This function extracts and displays the specific subsequences that constitute the
    detected pseudocycle, providing concentrated focus on the mathematical elements
    directly involved in the ω(mr) = mr verification process. The presentation isolates
    the pseudocycle from the broader sequence context, enabling detailed examination
    of the cycle structure and transformation pattern.
        
    Pseudocycle extraction and presentation:
        - p-parameters subsequence: Operation sequence for verification trace
        - m-parameters subsequence: Complete cycle from mr to mr in tuple-based domain
        
    Focused highlighting strategy:
        - p-subsequence: Emphasizes first and penultimate operations (verification endpoints)
        - m-subsequence: Highlights cycle boundaries (first and last mr occurrences)
        
    Mathematical significance of isolation:
        The subsequence presentation removes extraneous sequence elements, concentrating
        attention on the specific mathematical structure that defines the pseudocycle.
        This focused view facilitates understanding of the transformation pattern and
        prepares for detailed step-by-step verification analysis.
        
    Conditional display logic:
        Returns immediately without output if pseudocycle_length == 0, avoiding
        meaningless display of degenerate cycles and maintaining clean output flow
        for edge cases.
        
    Verification preparation:
        The displayed p-parameters subsequence corresponds exactly to the transformation
        sequence that will be applied during ω(mr) = mr verification, establishing
        direct visual connection between cycle structure and validation process.
        
    Side effects:
        - Writes pseudocycle subsequence analysis to stdout
        - Applies strategic highlighting for verification context
        - Provides focused mathematical structure presentation
        
    Output sections:
        [*] PSEUDOCYCLE BETWEEN TWO mr = {value} VALUES
            - p-parameters subsequence (with verification endpoint highlighting)
            - m-parameters subsequence (with cycle boundary highlighting)
        
    Note:
        This focused presentation bridges the gap between comprehensive sequence
        enumeration and detailed verification analysis, providing the specific
        mathematical context needed for ω(mr) = mr validation.
    """
    if data.pseudocycle_length == 0:
        return
    
    print(f"[*] PSEUDOCYCLE BETWEEN TWO mr = {data.mr_value} VALUES")
    print("")
    
    # p-parameters subsequence
    print("\t- p-parameters subsequence:")
    p_pseudocycle = get_p_pseudocycle(data)
    # Highlight first and penultimate p values (the ones used as first and last operations in verification)
    p_highlight_pos = (0, len(p_pseudocycle) - 2) if len(p_pseudocycle) >= 2 else (0,)
    formatted_p_pseudo = format_highlighted_sequence(p_pseudocycle, p_highlight_pos)
    print(f"\t{formatted_p_pseudo}")
    print("")
    
    # m-parameters subsequence
    print("\t- m-parameters subsequence:")
    m_pseudocycle = get_m_pseudocycle(data)
    m_highlight_pos = (0, len(m_pseudocycle) - 1)  # First and last mr positions
    formatted_m_pseudo = format_highlighted_sequence(m_pseudocycle, m_highlight_pos)
    print(f"\t{formatted_m_pseudo}")
    print("")


def display_verification_results(data: SequenceData, result: VerificationResult) -> None:
    """
    Present detailed step-by-step trace of the ω(mr) = mr verification process.
        
    This function generates a comprehensive tabular presentation of the mathematical
    verification process, displaying each transformation step in the sequence that
    tests whether the detected pseudocycle satisfies the fundamental condition
    ω(mr) = mr. The trace provides complete mathematical transparency and enables
    detailed analysis of verification success or failure.
        
    Verification trace structure:
        - Initial state: Starting mr value with "[Start]" indicator
        - Transformation sequence: Each T1/T2 operation with mathematical notation
        - Final verification: Comparison of result with original mr value
        - Status indicators: Success/failure assessment for each step
        
    Mathematical notation presentation:
        - T1 operations: "T1: 3 · current + 1" with precise mathematical symbols
        - T2 operations: "T2: ⌊current ÷ 2⌋" with floor function notation
        - Unicode mathematical symbols for professional presentation
        
    Error handling and overflow detection:
        Immediately reports verification failure if integer overflow is detected
        during the transformation process, providing the specific step number
        where computational limits were exceeded and preventing misleading results.
        
    Tabular format specifications:
        - Fixed-width columns for consistent alignment across variable data
        - Sequential step numbering for transformation traceability
        - Current value, operation type, and result for each transformation
        - Terminal status indicators for verification conclusion
        
    Side effects:
        - Writes complete verification trace to stdout
        - Displays formatted table with mathematical notation
        - Reports overflow conditions if computational limits exceeded
        
    Output sections:
        [*] TRACE OF ω(mr) = mr WITHIN THE PSEUDOCYCLE
            - Table header with column specifications
            - Initial state row with starting mr value
            - Sequential transformation steps with T1/T2 operations
            - Final verification status assessment
        
    Note:
        This detailed trace is essential for mathematical validation and provides
        the evidence needed to confirm or refute the pseudocycle hypothesis for
        the specific detected mr value and transformation sequence.
    """
    print("[*] TRACE OF ω(mr) = mr WITHIN THE PSEUDOCYCLE")
    print("")
    
    if result.overflow_step is not None:
        print(f"\tTransformation result: VERIFICATION FAILED (overflow at step {result.overflow_step})")
        return
    
    # Print table header
    print(format_verification_table_header())
    
    # Print initial step
    print(format_verification_step(1, 0, 0, 0, is_initial=True, mr_value=data.mr_value))
    
    # Print transformation steps
    for i, (current, p_val, next_val) in enumerate(result.transformation_steps):
        step = i + 2  # +2 because initial step is 1
        is_last = i == len(result.transformation_steps) - 1
        print(format_verification_step(step, current, p_val, next_val, 
                                     is_final=is_last, mr_value=data.mr_value))


def display_final_analysis(data: SequenceData, result: VerificationResult) -> None:
    """
    Present comprehensive mathematical summary and statistical analysis of pseudocycle verification.
        
    This function generates the definitive conclusion of the pseudocycle analysis,
    synthesizing verification results into a comprehensive mathematical assessment
    that includes both success/failure determination and detailed statistical
    breakdown of the transformation process. The analysis provides formal validation
    of the ω(mr) = mr condition and quantitative characterization of the pseudocycle.
        
    Verification conclusion presentation:
        - Success case: Formal mathematical notation confirming ω(mr) = mr
        - Failure case: Explicit contradiction showing ω(mr) ≠ mr with actual values
        - Transformation notation: T^n(mr) format indicating composition length
        
    Statistical analysis components:
        - T1 transformation count: Frequency of 3m+1 operations (odd Collatz cases)
        - T2 transformation count: Frequency of ⌊m/2⌋ operations (even Collatz cases)
        - Total transformation validation: Confirmation that T1+T2 = pseudocycle_length
        
    Mathematical significance interpretation:
        Success validation provides evidence for the mathematical consistency of
        the tuple-based transform approach and supports the theoretical framework
        underlying the pseudocycle analysis methodology.
        
    Formal notation standards:
        - ω(mr) = mr: Fundamental pseudocycle condition
        - T^n(mr): Composition notation for n sequential transformations
        - Statistical breakdown: Quantitative analysis of transformation types
        
    Verification completeness assessment:
        The analysis confirms that the total number of transformations equals
        the detected pseudocycle length, ensuring mathematical consistency and
        completeness of the verification process.
        
    Side effects:
        - Writes definitive mathematical conclusion to stdout
        - Displays formal verification results with statistical breakdown
        - Provides quantitative assessment of transformation composition
        
    Output sections:
        [*] ANALYSIS OF TUPLE-BASED TRANSFORM PSEUDOCYCLE
            - Verification success/failure determination
            - Formal mathematical notation (ω notation and T^n composition)
            - Statistical breakdown of T1/T2 transformation frequencies
            - Total transformation count validation
        
    Note:
        This final analysis represents the culmination of the mathematical
        investigation and provides the definitive assessment of whether the
        detected pseudocycle satisfies the theoretical predictions of the
        tuple-based transform framework.
    """
    print("")
    print("[*] ANALYSIS OF TUPLE-BASED TRANSFORM PSEUDOCYCLE")
    print("")
    
    if result.success:
        final_value = result.transformation_steps[-1][2] if result.transformation_steps else data.mr_value
        print(f"\t- Verification successful: pseudocycle from mr = {data.mr_value} to mr = {data.mr_value} validated")
        print(f"\t\tω({data.mr_value}) = {data.mr_value}")
        print(f"\t\tT^{data.pseudocycle_length}({data.mr_value}) = {final_value}")
        print(f"\t\tTransformation T1 was used {result.t1_count} times")
        print(f"\t\tTransformation T2 was used {result.t2_count} times")
        print(f"\t\tTotal number of transformations was {result.t1_count + result.t2_count}, equal to pseudocycle length, {data.pseudocycle_length}")
    else:
        final_value = result.transformation_steps[-1][2] if result.transformation_steps else data.mr_value
        print("\t- Verification failed")
        print(f"\tVerification: T^{data.pseudocycle_length}({data.mr_value}) = {final_value}")
    
    print("")


# ***********************************************************************************
# * 10. INPUT VALIDATION
# ***********************************************************************************

def validate_command_line_args(args: List[str]) -> int:
    """
    Perform comprehensive validation and parsing of command-line arguments with user guidance.
        
    This function implements robust input validation for the program's single required
    parameter (initial Collatz sequence value), providing comprehensive error handling,
    user guidance, and educational examples. The validation ensures mathematical
    preconditions are satisfied before initiating computationally intensive pseudocycle
    analysis.
        
    Validation hierarchy:
        1. Argument count verification: Ensures exactly one parameter provided
        2. Type conversion validation: Confirms input can be parsed as integer
        3. Mathematical domain validation: Verifies n ≥ 1 for valid Collatz sequences
        
    Error handling with educational guidance:
        - Usage instructions: Complete syntax specification with parameter description
        - Practical examples: Concrete demonstration using n=27 for user reference
        - Specific error messages: Targeted feedback for different failure modes
        
    Input domain constraints:
        The function enforces n ≥ 1 as required by Collatz sequence mathematical
        definition, preventing undefined behavior and ensuring meaningful analysis
        results for all validated inputs.
        
    User experience optimization:
        Provides immediate feedback with corrective guidance rather than generic
        error messages, enabling users to understand proper usage patterns and
        recover from input errors efficiently.
        
    Side effects and termination:
        - Writes usage instructions and examples to stdout for invalid inputs
        - Calls sys.exit(1) for validation failures (terminates with error status)
        - No side effects for successful validation (pure function behavior)
        
    Args:
        args: Complete command-line argument list including program name (sys.argv)
            
    Returns:
        int: Validated positive integer suitable for Collatz sequence initialization
        
    Validation cases:
        - Success: Returns validated integer n ≥ 1
        - Wrong argument count: Displays usage and exits
        - Non-integer input: Reports parsing error and exits  
        - Non-positive integer: Reports domain error and exits
        
    Note:
        This function serves as the critical input gateway, ensuring all subsequent
        mathematical analysis operates on well-defined, computationally safe parameters
        while providing excellent user experience through comprehensive error guidance.
    """
    if len(args) != 2:
        print("[*] USAGE:")
        print(f"\tpython3 {args[0]} <n>")
        print("\t\tn: Initial positive integer for Collatz sequence (mandatory)")
        print("")
        print("[*] EXAMPLE:")
        print(f"\tpython3 {args[0]} 27")
        print("")
        sys.exit(1)
    
    try:
        n = int(args[1])
    except ValueError:
        print(f"[*] ERROR: Invalid input '{args[1]}'. Please enter a valid integer.")
        print("")
        sys.exit(1)
    
    if n < 1:
        print(f"[*] ERROR: n must be >= 1. You entered: {n}")
        print("")
        sys.exit(1)
    
    return n


# ***********************************************************************************
# * 11. HIGH-LEVEL WORKFLOWS
# ***********************************************************************************

def analyze_collatz_pseudocycles(n: int, constants: Constants) -> Optional[SequenceData]:
    """
    Execute the complete pseudocycle detection and analysis pipeline for a given initial value.
        
    This function orchestrates the comprehensive mathematical analysis workflow that
    transforms a single integer input into complete pseudocycle characterization across
    both Collatz and tuple-based transform domains. The pipeline integrates sequence
    generation, domain transformation, and pseudocycle detection into a unified
    analytical framework.
        
    Analysis pipeline architecture:
        1. Multi-domain sequence generation: Produces Collatz, m-sequence, and p-sequence
        2. Pseudocycle detection: Identifies first repeated mr value and boundaries
        3. Mathematical parameter extraction: Computes cycle length and position data
        4. Cross-domain correspondence: Establishes Collatz-to-tuple-based mapping
        
    Mathematical integration:
        The function represents the complete transformation from elementary Collatz
        iteration to sophisticated pseudocycle mathematical characterization,
        bridging classical number theory with the novel tuple-based transform framework.
        
    Computational robustness:
        Integrates all safety mechanisms including overflow protection, sequence length
        limits, and convergence detection to ensure reliable analysis across diverse
        input values and mathematical behaviors.
        
    Result encapsulation:
        Returns comprehensive SequenceData object containing all mathematical parameters
        required for subsequent verification analysis, or None if pseudocycle detection
        fails due to computational limits or mathematical divergence.
        
    Args:
        n: Initial positive integer for Collatz sequence generation
        constants: Configuration object containing computational limits and safety parameters
            
    Returns:
        Optional[SequenceData]: Complete pseudocycle analysis results including:
            - All three fundamental sequences (Collatz, m, p)
            - Detected mr value and position information
            - Pseudocycle length and boundary parameters
            - Cross-domain correspondence data
            Returns None if no pseudocycle detected within computational limits
        
    Mathematical significance:
        Successful analysis provides the complete mathematical foundation required
        for rigorous verification of the ω(mr) = mr condition and validation of
        the tuple-based transform theoretical framework.

    Time Complexity: O(k) where k is the generated sequence length
    Space Complexity: O(k) for storing all sequences and analysis data

    Note:
        This function represents the core analytical engine that transforms elementary
        integer input into sophisticated mathematical characterization suitable for
        advanced pseudocycle verification and theoretical validation.
    """
    collatz_seq, m_seq, p_seq = generate_all_sequences(n, constants)
    return create_sequence_data(collatz_seq, m_seq, p_seq)


def display_analysis_results(data: SequenceData, result: VerificationResult) -> None:
    """
    Orchestrate comprehensive presentation of complete pseudocycle analysis and verification results.
        
    This function coordinates the sequential display of all analytical components,
    presenting the complete mathematical investigation from initial sequence generation
    through final verification conclusion. The orchestrated presentation ensures
    logical flow and comprehensive coverage of all analytical findings.
        
    Presentation orchestration sequence:
        1. Program identification and context establishment
        2. Pseudocycle boundary mathematical characterization
        3. Complete sequence enumeration with cross-domain correspondence
        4. Focused pseudocycle subsequence analysis
        5. Detailed verification trace with step-by-step validation
        6. Comprehensive mathematical conclusion and statistical summary
        
    Educational presentation structure:
        The sequential display builds mathematical understanding progressively,
        starting with fundamental parameters and advancing through increasingly
        sophisticated analytical components, culminating in definitive verification
        assessment.
        
    Comprehensive coverage strategy:
        Ensures all mathematical aspects of the analysis are presented with
        appropriate detail and visual emphasis, providing complete transparency
        of the analytical process and enabling thorough result interpretation.
        
    Visual integration:
        Coordinates highlighting and formatting across all display components
        to maintain visual consistency and mathematical emphasis throughout
        the complete presentation sequence.
        
    Side effects:
        - Writes complete analytical presentation to stdout
        - Coordinates all display functions in logical sequence
        - Ensures comprehensive coverage of all analytical components
        
    Args:
        data: Complete SequenceData containing all pseudocycle parameters
        result: VerificationResult containing validation outcomes and statistics
        
    Presentation components:
        - Program header and identification
        - Pseudocycle boundary characterization
        - Complete sequence enumeration
        - Focused pseudocycle analysis
        - Detailed verification trace
        - Comprehensive mathematical conclusion
        
    Note:
        This orchestration function ensures that users receive complete and
        comprehensible presentation of all analytical findings, supporting
        both mathematical validation and educational understanding of the
        tuple-based transform pseudocycle analysis methodology.
    """
    display_program_header()
    display_pseudocycle_boundaries(data)
    display_all_sequences(data)
    display_pseudocycle_sequences(data)
    display_verification_results(data, result)
    display_final_analysis(data, result)


def main() -> int:
    """
    Execute the complete pseudocycle analysis program with comprehensive error handling and result reporting.
        
    This function implements the top-level program execution framework, orchestrating
    the complete analytical pipeline from command-line input validation through final
    mathematical conclusion presentation. The main function ensures robust execution
    with appropriate error handling and provides standardized exit codes for
    integration with external systems.
        
    Program execution architecture:
        1. Configuration initialization with computational safety parameters
        2. User interface establishment through program header display
        3. Input validation and parsing with educational error guidance
        4. Mathematical analysis pipeline execution with overflow protection
        5. Verification process with detailed mathematical validation
        6. Comprehensive result presentation with complete analytical coverage
        
    Error handling strategy:
        Implements defensive programming principles with graceful degradation,
        providing meaningful error messages and appropriate exit codes for
        different failure modes while maintaining program stability and
        user experience quality.
        
    Computational safety integration:
        Coordinates all safety mechanisms including sequence length limits,
        overflow detection, and convergence validation to ensure reliable
        execution across diverse mathematical inputs and system constraints.
        
    User experience optimization:
        Provides immediate visual feedback through header display regardless
        of subsequent analysis success, ensuring users understand program
        execution status and receive appropriate guidance for error conditions.
        
    Return code specifications:
        - 0: Successful analysis completion with valid pseudocycle detection
        - 1: Analysis failure due to computational limits, input errors, or exceptions
        
    Side effects:
        - Writes complete program output to stdout including headers and results
        - May terminate program execution through sys.exit for input validation errors
        - Handles all exceptions with appropriate error reporting
        
    Program flow control:
        Implements fail-fast validation for input parameters while ensuring
        header display occurs before potential early termination, providing
        consistent user experience regardless of execution outcome.
        
    Integration considerations:
        Designed for both interactive terminal usage and automated execution
        environments, with appropriate exit codes and error handling for
        reliable integration with external analysis frameworks.
        
    Returns:
        int: Program exit status (0 for success, 1 for failure)
        
    Note:
        This main function represents the complete program lifecycle management,
        ensuring robust execution and comprehensive result reporting while
        maintaining mathematical rigor and computational safety throughout
        the entire pseudocycle analysis process.
    """
    constants = Constants()
    
    # Always display header first
    display_program_header()
    
    try:
        # Input validation
        n = validate_command_line_args(sys.argv)
        
        # Data analysis pipeline
        data = analyze_collatz_pseudocycles(n, constants)
        
        if not data:
            print("No mr found (overflow or limit reached)")
            return 1
        
        # Verification
        verification_result = verify_pseudocycle_transitions(data, constants)
        
        # Output generation (without header since it's already displayed)
        display_pseudocycle_boundaries(data)
        display_all_sequences(data)
        display_pseudocycle_sequences(data)
        display_verification_results(data, verification_result)
        display_final_analysis(data, verification_result)
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())