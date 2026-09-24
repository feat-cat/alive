/**
 * Ambient declaration for the platform-injected `@edgeone/pages-blob` module.
 *
 * The module is not present in node_modules locally: EdgeOne Makers injects it
 * at runtime inside Functions. Declaring the module here lets TS resolve it;
 * concrete API surfaces are augmented by the importing modules (e.g.
 * `agents/_blob-tools.ts`).
 */
declare module '@edgeone/pages-blob' {
  // Platform-injected module — API is augmented by importers.
  export {}
}
