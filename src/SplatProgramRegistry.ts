import type { CovSplatGenerator, GsplatGenerator, SplatGenerator } from "./SplatGenerator";
import type { DynoProgram } from "./dyno/program";

export enum SplatProgramMode {
  Packed,
  PackedCovariance,
  Extended,
  ExtendedCovariance,
}

/** Generation programs owned by one renderer, shared by its accumulator ring. */
export class SplatProgramRegistry {
  private programs = new Map<GsplatGenerator | CovSplatGenerator, Map<SplatProgramMode, DynoProgram>>();

  get(generator: GsplatGenerator | CovSplatGenerator, mode: SplatProgramMode, create: () => DynoProgram): DynoProgram {
    let modes = this.programs.get(generator);
    if (!modes) {
      modes = new Map();
      this.programs.set(generator, modes);
    }
    let program = modes.get(mode);
    if (!program) {
      program = create();
      modes.set(mode, program);
    }
    return program;
  }

  retain(nodes: readonly SplatGenerator[]) {
    const retained = new Set<GsplatGenerator | CovSplatGenerator>();
    for (const node of nodes) {
      if (node.generator) retained.add(node.generator);
      if (node.covGenerator) retained.add(node.covGenerator);
    }
    for (const [generator, modes] of this.programs) {
      if (!retained.has(generator)) {
        for (const program of modes.values()) program.dispose();
        this.programs.delete(generator);
      }
    }
  }

  dispose() {
    for (const modes of this.programs.values()) {
      for (const program of modes.values()) program.dispose();
    }
    this.programs.clear();
  }
}
