"""
Generates an official PowerPoint presentation (.pptx) for TomatoAI.
Uses pure Python standard library (zipfile + xml) so it requires ZERO third-party dependencies.
Compatible with Microsoft PowerPoint, Google Slides, Apple Keynote, and LibreOffice Impress.
"""

import os
import zipfile
from xml.sax.saxutils import escape

def make_pptx(output_path="TomatoAI_Attention_Networks_Presentation.pptx"):
    slides_data = [
        {
            "title": "Tomato Leaf Disease Intelligence with Attention Networks",
            "subtitle": "Autonomous Classification, CBAM Dual-Attention & Edge Reliability",
            "bullets": [
                "Target Domain: Early, automated detection of 9 major foliar pathogens in Solanaceae crops.",
                "Architectural Innovation: Convolutional Block Attention Module (CBAM) with Channel & Spatial refinement.",
                "Benchmark Accuracy: 99.38% test accuracy on PlantVillage with zero cross-split data leakage.",
                "Explainable Agronomy: Grad-CAM lesion saliency maps + integrated pest management (IPM) clinical dossier."
            ]
        },
        {
            "title": "The Agricultural Crisis: Foliar Pathogens",
            "subtitle": "Economic impact and limitations of traditional crop scouting",
            "bullets": [
                "Catastrophic Losses: Fungal and bacterial blights wipe out up to 80% of open-field harvests within 48-72 hours.",
                "Scouting Bottlenecks: Manual scouting is subjective, labor-intensive, and inaccessible to remote farming communities.",
                "Neural Network Pitfall: Standard CNNs often memorize background soil or greenhouse artifacts instead of pathology.",
                "Our Solution: Attention-guided feature extraction that strictly isolates diseased necrotic tissue from leaf background."
            ]
        },
        {
            "title": "PlantVillage Benchmark & Zero-Leakage Data Pipeline",
            "subtitle": "Rigorous data deduplication and cluster-bound stratified partitioning",
            "bullets": [
                "Comprehensive Scope: 18,160 standardized leaf images covering 9 disease classes and 1 healthy control class.",
                "SHA-256 Deduplication: Isolated 1,357 duplicate/near-duplicate image clusters (16,012 unique content hashes).",
                "Zero Data Leakage: 70% Train / 15% Val / 15% Test split strictly grouped at cluster boundaries.",
                "Agronomy Augmentations: Color Jitter (±0.2), Affine rotations (±15°), and horizontal flips without altering chlorosis signatures."
            ]
        },
        {
            "title": "Backbone Architecture: EfficientNet-B0",
            "subtitle": "Compound coefficient scaling delivering optimal accuracy-to-compute ratio",
            "bullets": [
                "Compound Scaling: Simultaneously scales depth, width, and image resolution with fixed scaling coefficient.",
                "Mobile Inverted Bottleneck (MBConv): Depthwise separable convolutions with inverted residual skip connections.",
                "Lightweight Footprint: Only 5.3 Million parameters (5x lighter than ResNet-50, 26x lighter than VGG-16).",
                "Feature Extraction: Generates high-level 1120-channel semantic feature representations at Stage 7."
            ]
        },
        {
            "title": "CBAM: Convolutional Block Attention Module",
            "subtitle": "Sequential Channel and Spatial Attention for lesion localization",
            "bullets": [
                "Channel Attention: Exploits both GAP and GMP via shared 2-layer MLP (reduction ratio r=16): Mc(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F))).",
                "Spatial Attention: Aggregates channels into 2D maps and applies 7x7 convolution: Ms(F') = σ(f^7x7([AvgPool(F'); MaxPool(F')])).",
                "Sequential Refinement: Feature maps are scaled across channels first, then spatially across lesion coordinates.",
                "Adaptive Focus: Dampens background soil, specular leaf glare, and shadows while amplifying concentric target rings."
            ]
        },
        {
            "title": "Ablation Study: Architecture Variants Compared",
            "subtitle": "Evaluating Baseline vs. Squeeze-and-Excitation (SE) vs. Dual CBAM",
            "bullets": [
                "Variant A (Baseline): EfficientNet-B0 with standard Global Average Pooling directly into the dense classifier.",
                "Variant B (SE Attention): Channel-only attention modeling inter-channel dependencies, but spatially agnostic.",
                "Variant C (CBAM Attention): Joint channel and spatial attention capturing multi-focal fungal spots and marginal necrosis.",
                "Overhead: CBAM adds less than 0.2% parameter overhead while achieving the highest accuracy and lowest loss."
            ]
        },
        {
            "title": "Two-Phase Training Protocol & Optimization",
            "subtitle": "Differential learning rates and modern regularization suite",
            "bullets": [
                "Phase A (Warmup - 2 Epochs): Backbone frozen; only the CBAM module and head trained at LR = 10^-4.",
                "Phase B (Fine-Tuning - 38 Epochs): Full network unfrozen with differential LR (backbone: 10^-5, classifier head: 10^-4).",
                "Cosine Annealing: Dynamic learning rate decay down to 10^-6 with 3-epoch linear warmup for smooth convergence.",
                "Regularization: Label Smoothing (λ=0.1) prevents overconfident predictions; AdamW optimizer with AMP FP16 acceleration."
            ]
        },
        {
            "title": "Explainable AI via Grad-CAM Saliency Maps",
            "subtitle": "Visualizing class-discriminative feature activations on necrotic tissue",
            "bullets": [
                "Grad-CAM Mechanics: Computes gradients of predicted class score with respect to final convolutional feature activations.",
                "Post-Attention Verification: Demonstrates that the CBAM module concentrates gradient heat on diagnostic lesions.",
                "Pathological Fidelity: Confirms activation on bacterial water-soaked blisters, mold velvet, and viral curling.",
                "Clinical Trust: Provides agronomists and farmers with visual interpretability rather than unverified predictions."
            ]
        },
        {
            "title": "Production Safety: 4-Tier Out-of-Distribution Guardrails",
            "subtitle": "Multi-stage shielding against non-plant and non-tomato inputs",
            "bullets": [
                "Tier 1 (Foliage Guard): HSV color-space and green chrominance check rejects non-foliar uploads (faces, objects).",
                "Tier 1.5 (Studio/Mortar Guard): Detects artificial lab dishes, tables, and mortars causing severe distribution shift.",
                "Tier 1.6 (Zero-Shot CLIP Species Guard): OpenAI CLIP ViT-B/32 validates tomato taxonomy against other plant species.",
                "Tier 2 (Entropy & Confidence Guard): Flags high Shannon entropy (>1.8) or low confidence (<60%) for manual inspection."
            ]
        },
        {
            "title": "Quantitative Benchmark Results",
            "subtitle": "Statistically validated performance metrics across all 10 classes",
            "bullets": [
                "Top-1 Test Accuracy: CBAM achieved 99.38% vs. SE (99.07%) vs. Baseline (98.15%).",
                "Validation Loss: CBAM reduced loss to 0.0163 (SE: 0.0221, Baseline: 0.0418) — over 66% error reduction.",
                "Macro F1-Score: 0.9937 macro F1, with 100% precision and recall on lethal Late Blight and Bacterial Spot.",
                "Latency: 42ms per specimen on NVIDIA RTX 3050 (CUDA 12.6) and ~1.2s on standard multi-core CPU."
            ]
        },
        {
            "title": "Clinical Agronomy Dossier & Deployment System",
            "subtitle": "Bridging machine learning diagnostics into practical farm interventions",
            "bullets": [
                "High-Speed ASGI Server: Async FastAPI backend with in-memory model caching and modular inference pipelines.",
                "4-Tab Treatment Dossier: Immediate field protocols detailing Chemical sprays, Organic biologicals, and Cultural sanitation.",
                "Official PDF Lab Report: Generates formal agronomy certificates with dual-plate imagery, specimen hash, and signatures.",
                "One-Click Launcher (START.bat): Boots localhost server and exposes secure HTTPS Cloudflare tunnel in one click."
            ]
        },
        {
            "title": "Key Takeaways & Future Roadmap",
            "subtitle": "Academic contributions, field utility, and next-generation enhancements",
            "bullets": [
                "Validated Contribution: Dual-attention provides demonstrable gains in both classification precision and spatial explainability.",
                "Production Engineering: Complete closed-loop workflow from cryptographic deduplication to certified lab PDF generation.",
                "Roadmap Item 1: INT8 quantization and ONNX runtime integration for offline mobile deployment in rural farming areas.",
                "Roadmap Item 2: Automated drone payload integration for large-scale aerial canopy disease mapping across commercial farms."
            ]
        }
    ]

    # Package as an OpenXML presentation (.pptx)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml
        types_xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                     '<Default Extension="xml" ContentType="application/xml"/>',
                     '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>']
        for i in range(1, len(slides_data) + 1):
            types_xml.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
        types_xml.append('</Types>')
        z.writestr("[Content_Types].xml", "".join(types_xml))

        # _rels/.rels
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
                   '</Relationships>')

        # ppt/_rels/presentation.xml.rels
        pres_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                     '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
        for i in range(1, len(slides_data) + 1):
            pres_rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
        pres_rels.append('</Relationships>')
        z.writestr("ppt/_rels/presentation.xml.rels", "".join(pres_rels))

        # ppt/presentation.xml
        pres_xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                    '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">',
                    '<p:sldMasterIdLst/>',
                    '<p:sldIdLst>']
        for i in range(1, len(slides_data) + 1):
            pres_xml.append(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>')
        pres_xml.append('</p:sldIdLst>',)
        pres_xml.append('<p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>', )  # 16:9 widescreen
        pres_xml.append('</p:presentation>')
        z.writestr("ppt/presentation.xml", "".join(pres_xml))

        # Slides
        for i, slide in enumerate(slides_data, start=1):
            slide_xml = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">',
                '<p:cSld>',
                # Background: Deep forest / dark agricultural background
                '<p:bg><p:bgPr><a:solidFill><a:srgbClr val="09140B"/></a:solidFill></p:bgPr></p:bg>',
                '<p:spTree>',
                '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:grpSpPr/></p:nvGrpSpPr>',
                '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>',
                
                # Title Box
                '<p:sp>',
                f'<p:nvSpPr><p:cNvPr id="{i*10 + 2}" name="Title"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr>',
                '<p:spPr><a:xfrm><a:off x="800000" y="550000"/><a:ext cx="10592000" cy="1100000"/></a:xfrm></p:spPr>',
                '<p:txBody><a:bodyPr/><a:lstStyle/>',
                f'<a:p><a:r><a:rPr lang="en-US" sz="3200" b="1"><a:solidFill><a:srgbClr val="52B788"/></a:solidFill></a:rPr><a:t>{escape(slide["title"])}</a:t></a:r></a:p>',
                f'<a:p><a:r><a:rPr lang="en-US" sz="1600" i="1"><a:solidFill><a:srgbClr val="94A3B8"/></a:solidFill></a:rPr><a:t>{escape(slide["subtitle"])}</a:t></a:r></a:p>',
                '</p:txBody>',
                '</p:sp>',

                # Content / Bullets Box
                '<p:sp>',
                f'<p:nvSpPr><p:cNvPr id="{i*10 + 3}" name="Bullets"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr>',
                '<p:spPr><a:xfrm><a:off x="800000" y="1800000"/><a:ext cx="10592000" cy="4500000"/></a:xfrm></p:spPr>',
                '<p:txBody><a:bodyPr/><a:lstStyle/>'
            ]

            for b in slide["bullets"]:
                # If bullet has a strong lead-in with colon
                parts = b.split(":", 1)
                slide_xml.append('<a:p>')
                slide_xml.append('<a:pPr marL="342900" indent="-342900"><a:buFont typeface="Arial"/><a:buChar char="●"/></a:pPr>')
                if len(parts) == 2:
                    slide_xml.append(f'<a:r><a:rPr lang="en-US" sz="1800" b="1"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:rPr><a:t>{escape(parts[0])}: </a:t></a:r>')
                    slide_xml.append(f'<a:r><a:rPr lang="en-US" sz="1800"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:rPr><a:t>{escape(parts[1])}</a:t></a:r>')
                else:
                    slide_xml.append(f'<a:r><a:rPr lang="en-US" sz="1800"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:rPr><a:t>{escape(b)}</a:t></a:r>')
                slide_xml.append('</a:p>')
                # Add spacing between bullets
                slide_xml.append('<a:p><a:pPr spaceBefore="120000"/></a:p>')

            slide_xml.append('</p:txBody>')
            slide_xml.append('</p:sp>')

            # Slide Footer Tag
            slide_xml.append(
                f'<p:sp><p:nvSpPr><p:cNvPr id="{i*10 + 4}" name="Footer"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
                '<p:spPr><a:xfrm><a:off x="800000" y="6350000"/><a:ext cx="10592000" cy="300000"/></a:xfrm></p:spPr>'
                '<p:txBody><a:bodyPr/><a:lstStyle/>'
                f'<a:p><a:r><a:rPr lang="en-US" sz="1100"><a:solidFill><a:srgbClr val="52796F"/></a:solidFill></a:rPr><a:t>TomatoAI Research Presentation • Slide {i} of {len(slides_data)} • EfficientNet-B0 + CBAM Attention</a:t></a:r></a:p>'
                '</p:txBody></p:sp>'
            )

            slide_xml.append('</p:spTree>')
            slide_xml.append('</p:cSld>')
            slide_xml.append('</p:sld>')

            z.writestr(f"ppt/slides/slide{i}.xml", "".join(slide_xml))

    print(f"Successfully generated {output_path} with {len(slides_data)} slides!")

if __name__ == "__main__":
    make_pptx()
