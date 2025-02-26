from pydantic import BaseModel, Field, RootModel
from typing import List, Union, Optional, Literal
from bloblang.input import CamferFeature, CamferSketchEntity, Constraint, DimensionConstraint, AngleDimensionConstraint, IdReference

### Command Arguments ###
class SetUnitsArgs(BaseModel):
    UnitName: Literal["mm", "cm", "m", "in", "ft", "A", "nm", "um", "mil", "uin"]

class AddFeatureArgs(BaseModel):
    Feature: CamferFeature

class AddSketchEntitiesArgs(BaseModel):
    Id: str
    Entities: List[CamferSketchEntity]

class AddSketchConstraintsArgs(BaseModel):
    Id: str
    Constraints: List[Union[Constraint, DimensionConstraint, AngleDimensionConstraint]]

class RemoveSketchEntitiesArgs(BaseModel):
    Id: str
    EntityIds: List[IdReference]

class RemoveSketchConstraintsArgs(BaseModel):
    Id: str
    ConstraintIds: List[IdReference]

class ProjectEntitiesToPlaneArgs(BaseModel):
    Id: str
    EntityIds: List[IdReference]

class SetBodiesToKeepArgs(BaseModel):
    Id: str
    BodyIds: Optional[List[IdReference]] = Field(default_factory=list)
    KeepAll: Optional[bool] = True

class GetSplitBodiesArgs(BaseModel):
    TrimSurfaces: List[IdReference]
    
class ViewEntitiesArgs(BaseModel):
    Id: str
    EntityType: Literal["Body", "Face", "Edge", "Vertex"]

class ViewSilhouetteArgs(BaseModel):
    Id: str

### Command Models ###
class Command(BaseModel):
    Name: str
    Args: dict = Field(default_factory=dict)

class SetUnitsCommand(Command):
    Name: Literal["SetUnits"]
    Args: SetUnitsArgs

class AddFeatureCommand(Command):
    Name: Literal["AddFeature"]
    Args: AddFeatureArgs

class AddSketchEntitiesCommand(Command):
    Name: Literal["AddSketchEntities"]
    Args: AddSketchEntitiesArgs
    
class AddSketchConstraintsCommand(Command):
    Name: Literal["AddSketchConstraints"]
    Args: AddSketchConstraintsArgs
    
class RemoveSketchEntitiesCommand(Command):
    Name: Literal["RemoveSketchEntities"]
    Args: RemoveSketchEntitiesArgs
    
class RemoveSketchConstraintsCommand(Command):
    Name: Literal["RemoveSketchConstraints"]
    Args: RemoveSketchConstraintsArgs

class ProjectEntitiesToPlaneCommand(Command):
    Name: Literal["ProjectEntitiesToPlane"]
    Args: ProjectEntitiesToPlaneArgs
    
class SetBodiesToKeepCommand(Command):
    Name: Literal["SetBodiesToKeep"]
    Args: SetBodiesToKeepArgs
    
class GetSplitBodiesCommand(Command):
    Name: Literal["GetSplitBodies"]
    Args: GetSplitBodiesArgs

class ViewEntitiesCommand(Command):
    Name: Literal["ViewEntities"]
    Args: ViewEntitiesArgs

class ViewSilhouetteCommand(Command):
    Name: Literal["ViewSilhouette"]
    Args: ViewSilhouetteArgs

class SubmitCommand(Command):
    Name: Literal["Submit"]
    Args: dict = Field(default_factory=dict)

class CamferCommand(RootModel):
    root: Union[
        SetUnitsCommand,
        AddFeatureCommand, 
        AddSketchEntitiesCommand, 
        AddSketchConstraintsCommand, 
        RemoveSketchEntitiesCommand, 
        RemoveSketchConstraintsCommand, 
        ProjectEntitiesToPlaneCommand, 
        SetBodiesToKeepCommand, 
        GetSplitBodiesCommand, 
        ViewEntitiesCommand, 
        ViewSilhouetteCommand,
        SubmitCommand
    ] = Field(..., discriminator='Name')
