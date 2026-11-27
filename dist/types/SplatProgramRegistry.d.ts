import { CovSplatGenerator, GsplatGenerator, SplatGenerator } from './SplatGenerator';
import { DynoProgram } from './dyno/program';
export declare enum SplatProgramMode {
    Packed = 0,
    PackedCovariance = 1,
    Extended = 2,
    ExtendedCovariance = 3
}
/** Generation programs owned by one renderer, shared by its accumulator ring. */
export declare class SplatProgramRegistry {
    private programs;
    get(generator: GsplatGenerator | CovSplatGenerator, mode: SplatProgramMode, create: () => DynoProgram): DynoProgram;
    retain(nodes: readonly SplatGenerator[]): void;
    dispose(): void;
}
