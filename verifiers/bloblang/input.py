from pydantic import BaseModel, ConfigDict, Field, RootModel
from typing import Literal, List, Optional, Union

### Reference models
class NothingReference(BaseModel):
    Type: Literal["None"] = "None"

class IdReference(BaseModel):
    Id: str

# SKETCHES #

### Sketch Entity models
class ChildPoint(BaseModel):
    X: float
    Y: float
    Z: float

class Entity(BaseModel):
    Type: str
    model_config = ConfigDict(extra='forbid') # necessary for strict validation

class Point(Entity):
    Type: Literal["Point"]
    X: float
    Y: float
    Z: float

class Line(Entity):
    Type: Literal["Line"]
    Construction: bool
    Start: ChildPoint
    End: ChildPoint

class Circle(Entity):
    Type: Literal["Circle"]
    Construction: bool
    Center: ChildPoint
    Radius: float

# class CenterPointArc(Entity):
#     Type: Literal["Arc"]
#     Construction: bool
#     Clockwise: bool
#     Start: ChildPoint
#     End: ChildPoint
#     Center: ChildPoint

class RadiusPointArc(Entity):
    Type: Literal["Arc"]
    Construction: bool
    Start: ChildPoint
    End: ChildPoint
    RadiusPoint: ChildPoint      

class Parabola(Entity):
    Type: Literal["Parabola"]
    Construction: bool
    Start: ChildPoint
    End: ChildPoint
    Apex: ChildPoint
    Focal: ChildPoint
    
class InterpolatedSplineHandleDefault(BaseModel):
    Type: Literal["Default"]

class InterpolatedSplineHandleTangent(BaseModel):
    Type: Literal["Tangent"]
    TangentAngle: float
    TangentMagnitudes: List[float] # Should be a float of size 2 (vector)

class InterpolatedSplineHandleCurvature(BaseModel):
    Type: Literal["Curvature"]

class InterpolatedSplinePoint(BaseModel):
    Point: ChildPoint
    Handle: Union[InterpolatedSplineHandleDefault, InterpolatedSplineHandleTangent, InterpolatedSplineHandleCurvature]

class InterpolatedSpline(Entity):
    Type: Literal["InterpolatedSpline"]
    Construction: bool
    SplinePoints: List[InterpolatedSplinePoint]
    Proportional: bool

class Polygon(Entity):
    Type: Literal["InputPolygon"]
    Construction: bool
    Center: ChildPoint
    FirstVertex: ChildPoint
    EdgeCount: int
    InscribedCircle: bool

class Ellipse(Entity):
    Type: Literal["InputEllipse"]
    Construction: bool
    Center: ChildPoint
    Major: ChildPoint
    Minor: ChildPoint

class EllipticalArc(Entity):
    Type: Literal["InputEllipticalArc"]
    Construction: bool
    Clockwise: bool
    Start: ChildPoint
    End: ChildPoint
    Center: ChildPoint
    Major: ChildPoint
    Minor: ChildPoint

class LinearPattern(Entity):
    Type: Literal["InputLinearPattern"]
    Seeds: List[IdReference] # list of entities to include
    DirectionX: List[float]
    DirectionY: List[float]
    CountX: int
    CountY: int

class CircularPattern(Entity):
    Type: Literal["InputCircularPattern"]
    Seeds: List[IdReference] # list of entities to include
    Center: ChildPoint
    AngleStep: float
    Count: int

### Constraint models
CONSTRAINT_TYPES = Literal[
    "Distance",
    "Angle",
    "Radius",
    "Horizontal",
    "Vertical",
    "Tangent",
    "Parallel",
    "Perpendicular",
    "Coincident",
    "Cocentric",
    "Symmetric",
    "Midpoint",
    "AtIntersect",
    "Equal",
    "Diameter",
    "OffsetEdge",
    "Fixed",
    "ArcAngle90",
    "ArcAngle180",
    "ArcAngle270",
    "ArcAngleTop",
    "ArcAngleBottom",
    "ArcAngleLeft",
    "ArcAngleRight",
    "HorizontalPoints",
    "VerticalPoints",
    "Collinear",
    "Coradial",
    "SnapGrid",
    "SnapLength",
    "SnapAngle",
    "UseEdge",
    "EllipseAngle90",
    "EllipseAngle180",
    "EllipseAngle270",
    "EllipseAngleTop",
    "EllipseAngleBottom",
    "EllipseAngleLeft",
    "EllipseAngleRight",
    "AtPierce",
    "DoubleDistance",
    "MergePoints",
    "Angle3Points",
    "ArcLength",
    "Normal",
    "NormalPoints",
    "SketchOffset",
    "AlongX",
    "AlongY",
    "AlongZ",
    "AlongXPoints",
    "AlongYPoints",
    "AlongZPoints",
    "ParallelYZ",
    "ParallelXZ",
    "Intersection",
    "Patterned",
    "IsoByPoint",
    "SameIsoParam",
    "FitSpline",
    "EqualCurvature",
    "EqualTangent",
    "TangentFace",
    "AlongX3D",
    "AlongY3D",
    "AlongXPoints3D",
    "AlongYPoints3D",
    "Traction",
    "BeltTraction",
    "BlockFixedLock",
    "BlockNormalLock",
    "BlockRotateLock",
    "FakeSlotConstraint",
    "FixedSlot",
    "SameSlot",
    "LinearPatternCount",
    "CircularPatternCount",
    "RadialOffset",
    "PlanarOffset",
    "EqualCurve3DAlign",
    "FlangeFaceDistance",
    "ConicRho",
    "C3Touch",
    "DoubleAngle",
    "SameCurveLength",
]

DIMENSION_CONSTRAINT_TYPES = Literal[
    "Ordinate",
    "Distance",
    "ArcLength",
    "Radial", 
    "Diameter",
    "HorizontalOrdinate",
    "VerticalOrdinate",
    "ZAxis",
    "Chamfer",
    "HorizontalDistance",
    "VerticalDistance",  
    "Scalar",
    "DoubleAngular",
    "DoubleDistance",
    "AngularOrdinate",
]

### Constraint models
class Constraint(BaseModel):
    References: List[IdReference]
    Type: CONSTRAINT_TYPES

class DimensionConstraint(BaseModel):
    References: List[IdReference]
    DimensionType: DIMENSION_CONSTRAINT_TYPES
    TextLocation: List[float]
    Value: Optional[float] = None

class AngleDimensionConstraint(BaseModel):
    References: List[IdReference]
    DimensionType: Literal["Angular"]
    TextLocation: List[float]
    Direction: Literal['None', 'Left', 'Right', 'Up', 'Down']
    Value: float
    Clockwise: bool
    UseOppExtLine: bool

# FEATURES #
class Feature(BaseModel):
    Type: str
    model_config = ConfigDict(extra='forbid') # necessary for strict validation

class FeatureScopeArgs(BaseModel):
    AutoSelectBodies: Optional[bool] = True
    Bodies: Optional[List[IdReference]] = Field(default_factory=list)

### Sketch Feature Model
class CamferSketchEntity(RootModel):
    root: Union[
        Point,
        Line,
        Circle,
        RadiusPointArc,
        Parabola,
        InterpolatedSpline,
        Polygon,
        Ellipse,
        EllipticalArc,
        LinearPattern,
        CircularPattern,
    ] = Field(..., discriminator='Type')

class SketchFeature(Feature):
    Type: Literal["Sketch"]
    ReferencePlane: IdReference
    Entities: List[CamferSketchEntity]
    Constraints: Optional[List[Union[Constraint, DimensionConstraint, AngleDimensionConstraint]]] = Field(default_factory=list)

### Extrude Feature Model
class EndConditionDistanceParams(BaseModel):
    Distance: float

class EndConditionDistance(BaseModel):
    EndConditionType: Literal["Blind", "MidPlane"]
    EndConditionParams: EndConditionDistanceParams
    
class EndConditionReferenceParams(BaseModel):
    End: IdReference

class EndConditionReference(BaseModel):
    EndConditionType: Literal["UpToVertex", "UpToSurface", "UpToBody", "UpToEntity"]
    EndConditionParams: EndConditionReferenceParams

class EndConditionOffsetFromParams(BaseModel):
    Distance: float
    TranslateSurface: bool
    Reversed: bool
    Surface: IdReference

class EndConditionOffset(BaseModel):
    EndConditionType: Literal["OffFrom"]
    EndConditionParams: EndConditionOffsetFromParams

class EndConditionThrough(BaseModel):
    EndConditionType: Literal["ThroughAll", "ThroughNext"]
    EndConditionParams: dict = Field(default_factory=dict) # Should be an empty dict

class ExtrudeEndCondition(RootModel):
    root: Union[
        EndConditionThrough, 
        EndConditionDistance, 
        EndConditionReference, 
        EndConditionOffset
    ] = Field(..., discriminator='EndConditionType')

class ExtrudeFeature(Feature):
    Type: Literal["Extrude"]
    Sketch: IdReference
    EndCondition: List[ExtrudeEndCondition]
    Reversed: Optional[bool] = False
    BothDir: Optional[bool] = False
    MergeWithAll: Optional[bool] = False
    DraftAngleA: Optional[float] = 0
    DraftAngleB: Optional[float] = 0
    OffsetDistance: Optional[float] = 0
    OffsetReversed: Optional[bool] = False
    Contours: Optional[List[IdReference]] = Field(default_factory=list)
    Start: Optional[Union[NothingReference, IdReference]] = NothingReference()
    StartType: Optional[Literal["Sketch", "Surface", "Vertex", "Offset"]] = "Sketch"

### Cut Extrude Feature Model
class CutExtrudeFeature(Feature):
    Type: Literal["CutExtrude"]
    Sketch: IdReference
    EndCondition: List[ExtrudeEndCondition]
    FlipSideToCut: Optional[bool] = False
    Reversed: Optional[bool] = False
    BothDir: Optional[bool] = False
    DraftAngleA: Optional[float] = 0
    DraftAngleB: Optional[float] = 0
    OffsetDistance: Optional[float] = 0
    OffsetReversed: Optional[bool] = False
    FeatureScope: Optional[FeatureScopeArgs] = FeatureScopeArgs()
    Contours: Optional[List[IdReference]] = Field(default_factory=list)
    Start: Optional[Union[NothingReference, IdReference]] = NothingReference()
    StartType: Optional[Literal["Sketch", "Surface", "Vertex", "Offset"]] = "Sketch"

### Chamfer Feature Model
class AngleDistanceChamferParams(BaseModel):
    ChamferType: Literal["AngleDistance"]
    Angle: float
    Distance: float
    Flipped: Optional[bool] = False
    TangentProp: Optional[bool] = False

class DistanceChamferParams(BaseModel):
    ChamferType: Literal["Distance"]
    DistanceA: float
    DistanceB: Optional[float] = 0
    TangentProp: Optional[bool] = False

class VertexChamferParams(BaseModel):
    ChamferType: Literal["Vertex"]
    DistanceA: float
    DistanceB: float
    DistanceC: float

class ChamferFeature(Feature):
    Type: Literal["Chamfer"]
    Entities: List[IdReference]
    TypeParams: Union[AngleDistanceChamferParams, DistanceChamferParams, VertexChamferParams]
    KeepFeatures: Optional[bool] = True

### Fillet Feature Model
PROFILE_TYPES = Literal["Circular", "ConicRho", "ConicRadius", "CurvatureContinuous"]

class RadiusParams(BaseModel):
    Type: Literal["Radius"]
    RadiusA: float
    RadiusB: float
    Symmetric: bool = False
    ProfileType: Optional[PROFILE_TYPES] = "Circular"
    ConicRhoOrRadius: Optional[float] = 0

class HoldLinesParams(BaseModel):
    Type: Literal["HoldLines"]
    HoldLines: List[IdReference]
    ProfileType: Optional[PROFILE_TYPES] = "Circular"

class ConstantRadiusParams(BaseModel):
    FilletType: Literal["ConstantRadius"]
    Entities: List[IdReference]
    RadiusParams: RadiusParams
    TangentProp: Optional[bool] = False
    RoundCorners: Optional[bool] = False
    OverflowType: Optional[Literal["Default", "KeepEdge", "KeepSurface"]] = "Default"

class FaceParams(BaseModel):
    FilletType: Literal["Face"]
    FaceSetA: List[IdReference]
    FaceSetB: List[IdReference]
    FaceTypeParams: Union[RadiusParams, HoldLinesParams] = Field(..., discriminator='Type')
    HelpPoint: Optional[Union[NothingReference, IdReference]] = NothingReference()
    TangentProp: Optional[bool] = False

class FullRoundParams(BaseModel):
    FilletType: Literal["FullRound"]
    FaceSetA: List[IdReference]
    FaceSetB: List[IdReference]
    FaceSetC: List[IdReference]
    TangentProp: Optional[bool] = False

class FilletFeature(Feature):
    Type: Literal["Fillet"]
    FilletParams: Union[ConstantRadiusParams, FaceParams, FullRoundParams] = Field(..., discriminator='FilletType')

### Revolve Feature Model
class RevolveFeature(Feature):
    Type: Literal["Revolve"]
    Axis: IdReference
    Sketch: IdReference
    AngleA: float
    AngleB: Optional[float] = 0
    Reversed: Optional[bool] = False
    MergeWithAll: Optional[bool] = False
    Contours: Optional[List[IdReference]] = Field(default_factory=list)

class CutRevolveFeature(Feature):
    Type: Literal["CutRevolve"]
    Sketch: IdReference
    Axis: IdReference
    AngleA: float
    AngleB: Optional[float] = 0
    Reversed: Optional[bool] = False
    FeatureScope: Optional[FeatureScopeArgs] = FeatureScopeArgs()
    Contours: Optional[List[IdReference]] = Field(default_factory=list)

### Sweep Feature Model
class CircularSweepParams(BaseModel):
    ProfileType: Literal["Circular"]
    Diameter: float

class SketchSweepParams(BaseModel):
    ProfileType: Literal["Sketch"]
    SketchProfile: IdReference
    Direction: Optional[Literal["DirectionA", "DirectionB", "Both"]] = "DirectionA"
    StartTangencyNormalToProfile: Optional[bool] = False
    EndTangencyNormalToProfile: Optional[bool] = False
    ProfileOrientation: Optional[Literal["FollowPath", "KeepNormalConstant"]] = "FollowPath"

class SweepFeature(Feature):
    Type: Literal["Sweep"]
    Path: IdReference
    ProfileParams: Union[CircularSweepParams, SketchSweepParams]
    MergeWithAll: Optional[bool] = False
    MergeTangentFaces: Optional[bool] = False

class CutSweepFeature(Feature):
    Type: Literal["CutSweep"]
    Path: IdReference
    ProfileParams: Union[CircularSweepParams, SketchSweepParams]
    MergeWithAll: Optional[bool] = False 
    MergeTangentFaces: Optional[bool] = False
    FeatureScope: Optional[FeatureScopeArgs] = FeatureScopeArgs()

### Helix Feature Model
class HelixFeature(Feature):
    Type: Literal["Helix"]
    Sketch: IdReference
    HelixType: Literal["PitchAndRevolution", "HeightAndRevolution", "HeightAndPitch", "Spiral"]
    Revolutions: float
    Pitch: float
    Height: float
    StartAngle: Optional[float] = 0
    Clockwise: Optional[bool] = False
    Reversed: Optional[bool] = False
    TaperAngle: Optional[float] = 0

### Circular Pattern Feature Model
class PatternBodyParams(BaseModel):
    SeedType: Literal["Body"]
    Bodies: List[IdReference]

class PatternFeatureParams(BaseModel):
    SeedType: Literal["Feature"]
    Features: List[IdReference]
    Geometry: Optional[bool] = True

class DistanceEndCondition(BaseModel):
    EndConditionType: Literal["Distance"]
    Distance: float

class CountEndCondition(BaseModel):
    EndConditionType: Literal["Count"]
    Count: int

class CircularPatternFeature(Feature):
    Type: Literal["CircularPattern"]
    SeedParams: Union[PatternBodyParams, PatternFeatureParams]
    Axis: IdReference
    EndConditionA: Union[DistanceEndCondition, CountEndCondition]
    EndConditionB: Optional[Union[DistanceEndCondition, CountEndCondition]] = None
    Deleted: Optional[List[int]] = Field(default_factory=list)
    Reversed: Optional[bool] = False
    VarySketch: Optional[bool] = False

### Linear Pattern Feature Model
class DistanceSpacingParams(BaseModel):
    SpacingType: Literal["Distance"]
    Distance: float

class CountSpacingParams(BaseModel):
    SpacingType: Literal["Count"]
    Count: int

class ReferenceEndCondition(BaseModel):
    EndConditionType: Literal["Reference"]
    Start: Optional[Union[NothingReference, IdReference]] = NothingReference()
    End: Optional[Union[NothingReference, IdReference]] = NothingReference()
    SpacingParams: Union[DistanceSpacingParams, CountSpacingParams]
    OffsetDistance: Optional[float] = 0
    ReversedOffset: Optional[bool] = False
    Reversed: Optional[bool] = False
    
class NormalEndCondition(BaseModel):
    EndConditionType: Literal["Normal"]
    Distance: float
    Count: int
    Reversed: Optional[bool] = False

class PatternLinearAxis(BaseModel):
    Axis: IdReference
    EndCondition: Union[NormalEndCondition, ReferenceEndCondition]

class LinearPatternFeature(Feature):
    Type: Literal["LinearPattern"]
    SeedParams: Union[PatternBodyParams, PatternFeatureParams]
    AxisA: PatternLinearAxis
    AxisB: Optional[PatternLinearAxis] = None
    VarySketch: Optional[bool] = False
    PatternSeedOnly: Optional[bool] = False
    Deleted: Optional[List[int]] = Field(default_factory=list)

### Combine Feature Model
class CombineFeature(Feature):
    Type: Literal["Combine"]
    Operation: Literal["Add", "Subtract", "Intersection"]
    Bodies: List[IdReference]

### Split Feature Model
class SplitFeature(Feature):
    Type: Literal["Split"]
    TrimSurfaces: List[IdReference]
    Bodies: List[IdReference]
    DeleteBodies: Optional[bool] = False

### Reference Axis Feature Model
class ReferenceAxisFeature(Feature):
    Type: Literal["ReferenceAxis"]
    References: List[IdReference]

### Reference Plane Feature Model
class ReferencePlaneConstraint(BaseModel):
    Type: Literal["Parallel", "Perpendicular", "Coincident", "Distance", "Angle", "Tangent", "Project", "MidPlane", "ParallelToScreen"]
    Selection: IdReference
    AngleOrDistance: Optional[float] = 0
    Flipped: Optional[bool] = False
    OriginOnCurve: Optional[bool] = False
    ProjectToNearestLocation: Optional[bool] = False

class ReferencePlaneFeature(Feature):
    Type: Literal["ReferencePlane"]
    FlipNormal: Optional[bool] = False
    Constraints: Optional[List[ReferencePlaneConstraint]] = Field(default_factory=list)

class CamferFeature(RootModel): 
    """A discriminated union of all possible features for schema validation"""
    root: Union[
        SketchFeature,
        ExtrudeFeature,
        CutExtrudeFeature,
        ChamferFeature,
        FilletFeature,
        RevolveFeature,
        CutRevolveFeature,
        SweepFeature,
        CutSweepFeature,
        HelixFeature,
        CircularPatternFeature,
        LinearPatternFeature,
        CombineFeature,
        SplitFeature,
        ReferenceAxisFeature,
        ReferencePlaneFeature,
    ] = Field(..., discriminator='Type')
