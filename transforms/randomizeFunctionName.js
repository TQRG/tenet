const utils = require('./lib/utils.js')

module.exports = function(fileInfo, api) {
  const j = api.jscodeshift
  utils.checkValidSource(fileInfo, j)

  let currentSource = j(fileInfo.source)
  const fnNames = {}

  // find all fn declarations
  currentSource.find(j.FunctionDeclaration)
    .forEach(p => {
      const currentName = p.value.id.name
      fnNames[currentName] = utils.randomAlphanumericString(12)
    })

  // for all identifiers, if they match a mapping
  // change the name
  currentSource.find(j.Identifier)
    .forEach(p => {
      const idName = p.value.name
      if (fnNames.hasOwnProperty(idName)) {
        p.value.name = fnNames[idName]
      }
      return p
    })
  
  return currentSource.toSource()
}