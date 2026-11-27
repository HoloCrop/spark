import { expect, test } from "./harness.fixture.js";

test("quartic rendering preserves the center while tightening the tail", async ({
  harnessPage,
}) => {
  const response = await harnessPage.evaluate(async () => {
    const h = window.harness;
    const THREE = h.three;
    const spark = h.createSpark({
      gaussianK: 1,
      preBlurAmount: 0,
      blurAmount: 0,
    });
    h.createCamera({ fov: 60, position: [0, 0, 3] });
    const splats = h.createPackedSplats({ maxSplats: 1 });
    splats.pushSplat(
      new THREE.Vector3(),
      new THREE.Vector3(0.2, 0.2, 0.1),
      new THREE.Quaternion(),
      0.8,
      new THREE.Color(1, 1, 1),
    );
    h.addSplatMesh({ packedSplats: splats });
    const sample = async () => {
      await h.settle();
      const gl = h.renderer.getContext();
      return [0, 8, 24].map((offset) => {
        const pixel = new Uint8Array(4);
        gl.readPixels(
          128 + offset,
          128,
          1,
          1,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          pixel,
        );
        return pixel[0];
      });
    };
    const gaussian = await sample();
    spark.gaussianK = 2;
    spark.setDirty();
    const quartic = await sample();
    return { gaussian, quartic };
  });
  expect(
    Math.abs(response.gaussian[0] - response.quartic[0]),
  ).toBeLessThanOrEqual(2);
  expect(response.quartic[1]).toBeGreaterThan(response.gaussian[1]);
  expect(response.quartic[2]).toBeLessThan(response.gaussian[2]);
});
