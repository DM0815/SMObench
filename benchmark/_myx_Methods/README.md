# Integration Methods

Original implementations of spatial multi-omics integration methods adapted for SMOBench evaluation.

## Available Methods

**CANDIES**
**COSMOS**
**PRAGA**
**PRESENT**
**SpaMV**
**SpaMosaic**
**SpatialGlue**
**SpaMultiVAE**

## Method Structure

Each method directory contains:
- Original method implementation files
- Adapted interface for SMOBench compatibility
- Method-specific configuration files
- Documentation and requirements

## Integration Support

**Vertical Integration**: All methods support cross-modality integration
**Horizontal Integration**: Method-dataset compatibility varies
**Mosaic Integration**: Selected methods support mixed scenarios

## Usage

Methods are executed through standardized scripts in the `Scripts/` directory rather than directly from this folder. This directory serves as the source repository for method implementations.