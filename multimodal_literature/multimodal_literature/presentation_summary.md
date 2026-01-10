# Multimodal Literature Review: Presentation Summary

## Overview
This document provides a comprehensive summary of 12 multimodal learning papers from the literature collection, organized by fusion strategies and architectural approaches. The collection includes both transformer-based and VAE-based methods, with particular relevance to multi-omics integration and genotype-guided drug design applications.

---

## 1. ALBEF: Aligning Latent Representations for Vision-Language Pre-training

### Key Technique
- **Useful technique before fusion**: ALBEF employs a strategy to align latent representations between vision and language modalities before the final fusion step.

### Architecture
- Pre-aligns vision and language embeddings in their respective latent spaces
- Facilitates better fusion by ensuring modalities are in compatible representation spaces
- Particularly effective as a preprocessing step for downstream multimodal fusion tasks

### Application Notes
- Can be used as a foundational alignment step in multi-omics projects
- Helps ensure that different data modalities (e.g., genomic, transcriptomic) are properly aligned before integration

---

## 2. BLIP-2: Bootstrapping Language-Image Pre-training

### Key Characteristics
- State-of-the-art vision-language model with advanced architecture
- **Status**: Potential utility unclear from initial review - requires further investigation for multi-omics applications

### Architecture
- Bootstrap approach to learning from noisy image-text pairs
- Effective for large-scale pretraining scenarios
- May have applicability for learning from noisy multi-omics data pairs

### Application Notes
- Architecture complexity may make it challenging to adapt for multi-omics
- Further research needed to determine specific use cases in genotype-induced drug design

---

## 3. CLAP: Contrastive Language-Audio Pre-training

### Key Fusion Strategy
- **Linear projection fusion**: Uses linear projection layers to map audio and text representations into a shared latent space
- Simpler fusion mechanism compared to transformer-based approaches

### Architecture
- Contrastive learning framework similar to CLIP
- Linear transformations to align modalities
- Efficient and interpretable fusion method

### Application Notes
- **Highly relevant**: The linear projection approach could be directly applicable to multi-omics fusion
- Can fuse genomic, transcriptomic, or proteomic data with clinical text data
- Simpler architecture makes it easier to interpret and adapt

---

## 4. CLIP: Contrastive Language-Image Pre-training (OpenAI)

### Key Characteristics
- **Simple supervised learning approach**: Uses contrastive learning with image-text pairs
- **Large storage requirements**: Requires extensive storage for image and text embeddings

### Architecture
- Dual-encoder architecture (separate encoders for each modality)
- Contrastive loss (InfoNCE) to align representations
- Simple but effective for zero-shot tasks

### Strengths
- Conceptually straightforward
- Excellent zero-shot generalization capabilities
- Well-established baseline in multimodal learning

### Limitations
- Requires large-scale pretraining data
- High storage and memory requirements
- May not capture complex cross-modal interactions

### Application Notes
- Can serve as a baseline for multi-omics contrastive learning
- May require adaptation for biological data characteristics

---

## 5. Flamingo: Few-Shot Learning with Frozen Pretrained Language Models

### Key Architecture
- **Transformer-based with gated cross-attention**: Uses transformer architecture with specialized gated cross-attention mechanisms
- **Knowledge incorporation**: Gated cross-attention allows incorporation of knowledge from other latent spaces

### Innovation
- Can work with frozen pretrained language models
- Gated mechanism controls information flow between modalities
- Effective few-shot learning capabilities

### Architecture Details
- Interleaves frozen pretrained language model layers with learnable cross-attention layers
- Gating mechanism prevents catastrophic forgetting
- Allows efficient incorporation of visual information into language models

### Application Notes
- **Highly relevant**: Gated cross-attention could be adapted for multi-omics integration
- Useful when one modality (e.g., genotype) needs to guide another (e.g., drug design)
- Allows preserving pretrained knowledge while incorporating new modalities

---

## 6. ImageBind: One Embedding Space for All Modalities

### Key Innovation
- **Multimodal embedding space**: Creates a unified embedding space for multiple modalities (audio, text, image, video, 3D, etc.)
- **Progressive alignment strategy**: Uses preliminary contrastive learning to align pairs of modalities, then uses InfoNCE loss to bind pairs and generate new modality pairs

### Architecture
- Starts with image-text alignment (as in CLIP)
- Progressively adds new modalities by aligning them with existing aligned pairs
- Leverages the transitive property of embedding spaces

### Application Strategy
- Preliminary contrastive learning aligns a pair of modalities (e.g., genotype-transcriptome)
- InfoNCE loss binds pairs of modalities
- Can generate new modality pairs through the unified embedding space

### Application Notes
- **Extremely relevant**: Progressive alignment strategy is perfect for multi-omics
- Can start with one omics pair and progressively add more modalities
- Unified embedding space enables cross-modal retrieval and generation

---

## 7. LLaVA: Large Language and Vision Assistant

### Key Architecture
- **Completely transformer-based**: Pure transformer architecture throughout
- **Context provision**: Essentially provides visual context to language models for accurate responses
- **Simple encoding and projection**: Uses straightforward encoding and projection layers rather than complex fusion mechanisms

### Design Philosophy
- Treats vision as context for language generation
- Simple representation learning followed by projection to LLM space
- Focuses on interpretability through simplicity

### Architecture Details
- Vision encoder (e.g., CLIP visual encoder)
- Linear projection to align vision features with LLM token space
- Language model processes both text and projected vision tokens

### Application Notes
- Could be adapted where one omics modality (e.g., genotype) provides context for another (e.g., drug design)
- Simpler architecture may be easier to interpret for biological applications
- Transformer-based approach is familiar and well-understood

---

## 8. Multimodal Collapse: Theoretical Framework

### Key Finding
**"Modality collapse happens when noisy features from one modality are entangled, via a shared set of neurons in the fusion head, with predictive features from another, effectively masking out positive contributions from the predictive features of the former modality and leading to its collapse."**

### Critical Insight
- **Theoretical establishment**: This paper provides the theoretical foundation for understanding modality collapse
- **Problem identification**: Explains why some modalities become ineffective in fusion networks
- **Mechanism**: Shared neurons in fusion head cause entanglement of noisy and predictive features

### Implications
- Fusion architectures must be designed to prevent this entanglement
- Need for regularization or architectural constraints to prevent modality collapse
- Important consideration for multi-omics integration where data quality varies across modalities

### Application Notes
- **Critical reading**: Essential theoretical understanding for multi-omics fusion
- Must design fusion heads to prevent modality collapse
- Consider techniques to separate noisy vs. predictive features
- Important for handling variable-quality omics data

---

## 9. PaLM-E: An Embodied Multimodal Language Model

### Architecture Notes
- **Complex architecture**: Architecture details are not entirely clear from initial review
- Embodied AI model combining language, vision, and robotics

### Characteristics
- Based on PaLM (Pathways Language Model)
- Incorporates continuous representations (images, robot states)
- Embodied reasoning capabilities

### Application Notes
- Architecture complexity may limit direct applicability
- Requires further investigation for multi-omics applications
- May have insights on handling continuous and discrete modalities together

---

## 10. ViLBERT: Vision-and-Language BERT

### Key Architecture
- **Co-attentional transformer layers**: Uses co-attention mechanisms where both modalities attend to each other simultaneously
- **Masked learning**: Employs masked learning objectives similar to BERT

### Innovation
- Separate streams for each modality
- Co-attention layers allow bidirectional cross-modal attention
- Masked multi-modal modeling for pretraining

### Architecture Details
- Two parallel streams (vision and language) with co-attentional transformer layers
- Masked language modeling and masked region modeling objectives
- Co-attention enables rich cross-modal interactions

### Application Notes
- **Highly relevant**: Co-attentional layers could work well for multi-omics
- Allows bidirectional information flow between modalities (e.g., genotype ↔ transcriptome)
- Masked learning could help with missing data scenarios common in omics
- Separate streams preserve modality-specific representations

---

## 11. Multimodal VAEs: Bridging Language, Vision and Action

### Key Innovation
- **Probabilistic multimodal fusion**: Extends Variational Autoencoders (VAEs) to handle multiple modalities through joint latent representations
- **Cross-generation capability**: Enables reconstructing one modality from another through shared latent space

### Architecture Variants

#### MVAE (Product of Experts)
- **Key Equation**: $q_\phi(z|x_1,...,x_N) = \prod_n q_\phi(z|x_n)$
- Assumes modalities are conditionally independent given latent $z$
- Product of individual expert distributions (Gaussian)
- **Strengths**: Clean probabilistic formulation, good when modalities are complementary
- **Limitations**: Requires all modalities present during training

#### MMVAE (Mixture of Experts)
- **Key Equation**: $q_\phi(z|x_1,...,x_N) = \sum_n \alpha_n q_\phi(z|x_n)$, where $\alpha_n = 1/N$
- Assumes all inputs have comparable complexity
- Mixture of unimodal posteriors
- **Strengths**: More flexible, can handle missing modalities
- **Limitations**: Requires Importance-Weighted Autoencoder (IWAE) objective for best performance

#### MoPoE (Mixture of Products of Experts)
- **Key Equation**: $q_{MoPoE}(z|X) = \frac{1}{2^N} \sum_{X_k \in P(X)} q_{PoE}(z|X_k)$
- Generalization of both MVAE and MMVAE
- Considers all possible subsets of modalities (powerset)
- **Strengths**: Most flexible, handles any combination of missing modalities
- **Limitations**: Computational complexity grows exponentially with number of modalities

### Mathematical Foundation
- **ELBO Objective**: $\mathcal{L}_{ELBO}(\phi, \theta) = \mathbb{E}_{q_\phi(z|x_i)}[\log p_\theta(x_i|z)] - KL(q_\phi(z|x_i)||p(z))$
  - First term: Reconstruction loss (negative log-likelihood)
  - Second term: KL divergence regularization (encourages latent to match prior, typically Gaussian)

### Application Notes
- **Extremely relevant for multi-omics**: VAE approaches are directly applicable to biological data
- **Current project uses**: PVAE (Profile VAE) for transcriptomics + SVAE (SMILES VAE) for molecules
- **Integration strategy**: Gaussian addition of latent spaces ($z = z_p + z_c$)
- Can adapt MVAE/MMVAE/MoPoE for more sophisticated multi-omics fusion
- Handles missing data naturally through probabilistic framework
- Enables cross-generation: e.g., generate molecules from transcriptomic profiles

---

## 12. Cross-Modal Variational Alignment of Latent Spaces

### Key Innovation
- **Variational alignment**: Aligns latent representations across modalities using variational inference principles
- **Unified latent space**: Creates a shared embedding space where aligned modalities can be compared and combined

### Architecture
- Uses variational inference to align latent distributions from different modalities
- Aligns posterior distributions $q_\phi(z|x_i)$ across modalities $i$
- Ensures statistical compatibility of latent representations

### Alignment Strategy
- Learns mappings that align latent spaces probabilistically
- Maintains distributional properties while enabling cross-modal operations
- Can be applied before fusion to ensure compatibility (similar to ALBEF pre-alignment)

### Application Notes
- **Direct relevance**: Your project uses Gaussian addition for latent alignment ($z = z_p + z_c$)
- **Theoretical foundation**: Provides principled approach to latent space alignment
- **Can improve current approach**: Instead of simple addition, use variational alignment for better statistical properties
- **Pre-fusion alignment**: Useful before applying fusion strategies (complements ALBEF approach)
- **Ensures compatibility**: Makes latent spaces from different modalities compatible for fusion operations
- **Critical for multi-omics**: Different omics data types have different statistical properties that need proper alignment

### Connection to Current Work
- Your genotype-guided model combines: $z = z_p + z_c$ where $z_p \sim \mathcal{N}(\mu_p, \sigma_p^2)$ and $z_c \sim \mathcal{N}(\mu_c, \sigma_c^2)$
- This paper provides theoretical justification and improved methods for such alignment
- Could enhance the integration mechanism in PVAE+SVAE architecture

---

## Integration with Current Project: PVAE + SVAE Architecture

### Current Approach Analysis

Your genotype-induced drug design project uses:
- **PVAE (Profile VAE)**: Encodes transcriptomic profiles $x_p \in \mathbb{R}^G$ to latent $z_p \sim \mathcal{N}(\mu_p, \sigma_p^2)$
- **SVAE (SMILES VAE)**: Encodes molecular SMILES to latent $z_c \sim \mathcal{N}(\mu_c, \sigma_c^2)$
- **Integration**: Simple Gaussian addition $z = z_p + z_c$

### How the New Papers Can Enhance Your Approach

#### From "Bridging Language, Vision and Action: Multimodal VAEs"

**Option 1: Upgrade to MVAE (Product of Experts)**
- Replace Gaussian addition with Product of Experts
- Joint posterior: $q(z|x_p, x_c) \propto q_p(z|x_p) \cdot q_c(z|x_c)$
- **Benefits**: More principled fusion, better captures joint distribution
- **Implementation**: Instead of $z = z_p + z_c$, sample from product distribution

**Option 2: Upgrade to MMVAE (Mixture of Experts)**
- If transcriptomic and molecular data have different complexities
- Joint posterior: $q(z|x_p, x_c) = \frac{1}{2}q_p(z|x_p) + \frac{1}{2}q_c(z|x_c)$
- **Benefits**: Handles varying data quality, can work with missing modalities
- **Use case**: When some samples have only genotype or only molecule data

**Option 3: Enable Cross-Generation**
- Current: Generate molecules from combined latent $z$
- Enhancement: Enable generating transcriptomes from molecules ($p(x_p|z_c)$)
- **Benefit**: Bidirectional generation, useful for drug repurposing

#### From "Cross-Modal Variational Alignment of Latent Spaces"

**Improve Latent Space Alignment**
- Current: Simple addition assumes aligned spaces
- Enhancement: Use variational alignment before addition
- **Method**: Learn alignment mapping $T$ such that $z = T(z_p) + z_c$ or better alignment
- **Benefit**: Ensures statistical compatibility, improves generation quality

### Recommended Upgrade Path

1. **Phase 1**: Keep current architecture, add variational alignment preprocessing
   - Implement Cross-Modal Variational Alignment between $z_p$ and $z_c$ spaces
   - Then perform addition: $z = \text{align}(z_p) + z_c$

2. **Phase 2**: Replace addition with MVAE Product of Experts
   - More principled than addition
   - Maintains probabilistic interpretation
   - Better theoretical foundation

3. **Phase 3**: Add cross-generation capability
   - Decode transcriptomes from molecule latents
   - Enable bidirectional generation
   - Useful for exploring genotype-drug relationships

4. **Phase 4**: Scale to more modalities (if needed)
   - Add epigenomic, proteomic data
   - Use MoPoE for flexible missing-modality handling
   - Leverage ImageBind-style progressive alignment strategy

---

## Comparative Analysis

### Fusion Strategies Summary

| Paper | Fusion Strategy | Complexity | Applicability to Multi-Omics |
|-------|----------------|------------|----------------------------|
| **ALBEF** | Pre-alignment before fusion | Medium | High - Useful preprocessing step |
| **BLIP-2** | Complex bootstrap learning | High | Medium - Requires investigation |
| **CLAP** | Linear projection | Low | **Very High** - Simple, interpretable |
| **CLIP** | Contrastive learning | Low | High - Good baseline |
| **Flamingo** | Gated cross-attention | Medium-High | **Very High** - Guided fusion |
| **ImageBind** | Progressive alignment | Medium | **Very High** - Scalable to many modalities |
| **LLaVA** | Simple projection | Low | Medium - Context provision |
| **Multimodal Collapse** | Theoretical framework | N/A | **Critical** - Design consideration |
| **PaLM-E** | Complex embodied model | High | Low - Unclear architecture |
| **ViLBERT** | Co-attentional layers | Medium | **Very High** - Bidirectional fusion |
| **Multimodal VAEs** (MVAE/MMVAE/MoPoE) | Probabilistic joint latent space | Medium-High | **Extremely High** - Directly applicable, handles missing data |
| **Cross-Modal Variational Alignment** | Variational latent alignment | Medium | **Extremely High** - Theoretical foundation for current work |

---

## Recommendations for Multi-Omics Applications

### Most Promising Approaches

1. **Multimodal VAEs** (MVAE/MMVAE/MoPoE) - **Top Priority for VAE-based Projects**
   - **Directly applicable**: Your project already uses VAEs (PVAE + SVAE)
   - **Handles missing data**: Probabilistic framework naturally handles missing modalities
   - **Cross-generation**: Can generate one modality from another (e.g., molecules from transcriptomes)
   - **Product of Experts (MVAE)**: Good for complementary modalities (genotype + transcriptome)
   - **Mixture of Experts (MMVAE)**: Better when modalities have varying complexity
   - **MoPoE**: Most flexible, handles any combination of missing modalities (scalable to many omics types)

2. **Cross-Modal Variational Alignment** - **Critical for Latent Space Integration**
   - **Theoretical foundation**: Provides principled approach for aligning latent spaces
   - **Improves current approach**: Enhances your Gaussian addition method ($z = z_p + z_c$)
   - **Pre-fusion alignment**: Should be considered before fusion operations
   - **Statistical compatibility**: Ensures latent distributions are properly aligned across modalities

3. **ImageBind** (Progressive Alignment)
   - Best for integrating many omics modalities progressively
   - Unified embedding space enables cross-modal operations
   - Can start small and scale up
   - Complementary to VAE approaches (can use ImageBind-style progressive alignment with VAE backbone)

4. **ViLBERT** (Co-Attentional Layers)
   - Excellent for bidirectional information flow
   - Handles missing data well through masked learning
   - Preserves modality-specific features
   - Can be combined with VAE encoders for hybrid architectures

5. **CLAP** (Linear Projection)
   - Simplest to implement and interpret
   - Fast training and inference
   - Good baseline for initial experiments
   - Can be used as final projection layer after VAE encoding

6. **Flamingo** (Gated Cross-Attention)
   - Ideal when one modality should guide another
   - Useful for genotype-guided drug design
   - Preserves pretrained knowledge
   - Can be adapted for VAE-based architectures

### Critical Considerations

1. **Multimodal Collapse**
   - Must design architectures to prevent modality collapse
   - Consider separate pathways for different modalities
   - Regularize fusion layers appropriately
   - **VAE approaches**: The probabilistic framework with KL regularization naturally helps prevent collapse

2. **Latent Space Alignment (Cross-Modal Variational Alignment)**
   - **Critical for VAE-based approaches**: Must properly align latent spaces before fusion
   - Current Gaussian addition ($z = z_p + z_c$) is simple but can be improved with variational alignment
   - Ensures statistical compatibility across modalities
   - Should be considered as preprocessing step (complements ALBEF approach)

3. **Handling Missing Modalities**
   - Multi-omics data often has missing modalities per sample
   - **MMVAE/MoPoE**: Explicitly designed for missing modalities
   - **MVAE**: Requires all modalities present (may need data imputation)
   - Consider computational complexity: MoPoE scales exponentially with number of modalities

4. **Pre-alignment (ALBEF + Variational Alignment)**
   - Consider aligning modalities before fusion
   - Variational alignment provides probabilistic pre-alignment
   - Can improve fusion quality significantly
   - Combines well with other fusion strategies

5. **Simplicity vs. Complexity**
   - Start with simpler approaches (CLAP, CLIP, or simple VAE with Gaussian addition)
   - Progress to more complex architectures (MMVAE, MoPoE, ViLBERT) as needed
   - Balance between expressiveness and interpretability
   - **VAE benefit**: Probabilistic framework provides interpretability through latent distributions

---

## Future Directions

1. **Hybrid Approaches**: Combine elements from multiple papers
   - **VAE + Transformer**: Use VAE encoders (PVAE, SVAE) with transformer fusion (ViLBERT co-attention)
   - **VAE + Progressive Alignment**: ImageBind-style progressive alignment with VAE backbone
   - **Variational Alignment + ViLBERT**: Pre-align latent spaces (Cross-Modal Variational Alignment) then use co-attention
   - **ALBEF pre-alignment + VAE fusion**: Pre-align with ALBEF, then use MMVAE/MoPoE for fusion
   - **Gated Cross-Attention + VAE**: Flamingo-style gating for genotype-guided fusion with VAE latent spaces

2. **VAE Architecture Selection for Multi-Omics**
   - **Start with MVAE (Product of Experts)**: If all modalities available and complementary (e.g., genotype + transcriptome)
   - **Upgrade to MMVAE (Mixture of Experts)**: If modalities have varying complexity or quality
   - **Use MoPoE**: When need to handle many modalities with frequent missing data (but watch computational cost)
   - **Consider hybrid**: Combine different VAE approaches (e.g., MVAE for certain modality pairs, MoPoE for full integration)

3. **Improving Current VAE-Based Approach**
   - **Replace Gaussian addition** ($z = z_p + z_c$) with **variational alignment** from Cross-Modal Variational Alignment paper
   - **Explore MMVAE/MoPoE** for more sophisticated fusion than simple addition
   - **Add cross-generation capability**: Enable generating transcriptomes from molecules and vice versa
   - **Handle missing data**: Adapt MMVAE/MoPoE for scenarios where not all omics types are available per sample

4. **Domain Adaptation**: Adapt vision-language architectures for biological data
   - **VAE advantage**: Probabilistic framework naturally handles biological variability
   - Consider biological priors and constraints in latent space design
   - Handle domain-specific challenges (missing data, batch effects, normalization)
   - Use KL regularization to encourage biologically meaningful latent representations

5. **Interpretability**: Focus on interpretable fusion mechanisms
   - **VAE interpretability**: Latent distributions provide uncertainty estimates
   - **Linear projections**: More interpretable (CLAP, simple projections)
   - **Attention weights**: Provide insights into cross-modal interactions (ViLBERT, Flamingo)
   - **Product/Mixture weights**: In MMVAE/MoPoE, weight distributions show modality importance

6. **Theoretical Understanding**: Build on multimodal collapse theory
   - Design architectures that prevent collapse
   - **VAE regularization**: KL divergence naturally helps prevent collapse
   - Understand when and why modalities become ineffective
   - Combine theoretical insights from multimodal collapse with VAE probabilistic framework

---

## References

Papers covered in this summary:
1. ALBEF: Aligning Latent Representations for Vision-Language Pre-training
2. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
3. CLAP: Learning Audio Concepts from Natural Language Supervision
4. CLIP: Learning Transferable Visual Models From Natural Language Supervision
5. Flamingo: A Visual Language Model for Few-Shot Learning
6. ImageBind: One Embedding Space To Bind Them All
7. LLaVA: Large Language and Vision Assistant
8. Multimodal Collapse: Theoretical framework for modality collapse
9. PaLM-E: An Embodied Multimodal Language Model
10. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks
11. Bridging Language, Vision and Action: Multimodal VAEs in Robotic Manipulation Tasks (2024)
12. Cross-modal Variational Alignment of Latent Spaces (Theodoridis et al., CVPRW 2020)

---

## Presentation Outline

### Slide Structure Recommendation

1. **Introduction** (1 slide)
   - Overview of multimodal learning challenges
   - Relevance to multi-omics integration
   - Current project context (PVAE + SVAE approach)

2. **Fusion Strategies Overview** (1 slide)
   - Different approaches to modality fusion
   - Taxonomy of methods: Transformer-based vs. VAE-based
   - Key distinction: Discriminative vs. Generative approaches

3. **Deep Dive: VAE-Based Approaches** (2 slides)
   - **Slide 3a**: Multimodal VAEs (MVAE, MMVAE, MoPoE)
     - Product vs. Mixture of Experts
     - Probabilistic fusion framework
     - Direct applicability to current work
   - **Slide 3b**: Cross-Modal Variational Alignment
     - Latent space alignment theory
     - Connection to current Gaussian addition approach
     - Improvement opportunities

4. **Deep Dive: Transformer-Based Approaches** (3 slides)
   - **Slide 4a**: ImageBind (Progressive Alignment) + ViLBERT (Co-Attentional Layers)
   - **Slide 4b**: CLAP (Linear Projection) + Flamingo (Gated Cross-Attention)
   - **Slide 4c**: ALBEF (Pre-alignment) - connection to variational alignment

5. **Critical Theory** (1 slide)
   - Multimodal Collapse phenomenon
   - How VAE regularization helps prevent collapse
   - Implications for architecture design

6. **Comparison & Recommendations** (1 slide)
   - Table comparing approaches
   - Best practices for multi-omics
   - Specific recommendations for VAE-based projects

7. **Hybrid Approaches & Future Work** (1 slide)
   - VAE + Transformer combinations
   - Improving current approach with variational alignment
   - Domain-specific adaptations

**Total: ~10-11 slides** (excluding title and references)

### Alternative Condensed Structure (8 slides)

1. **Introduction** (1 slide)
2. **Two Paradigms: VAE vs. Transformer** (1 slide)
3. **VAE Approaches for Multi-Omics** (1 slide - MVAE/MMVAE/MoPoE + Variational Alignment)
4. **Transformer Approaches** (1 slide - Top 4 methods)
5. **Multimodal Collapse Theory** (1 slide)
6. **Comparison & Recommendations** (1 slide)
7. **Current Work Integration** (1 slide - How to improve PVAE+SVAE)
8. **Future Directions** (1 slide)
