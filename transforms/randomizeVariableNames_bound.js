const utils = require('./lib/utils.js')

function isContained(fnLoc, targetLoc) {
    const {sline, scol, eline, ecol} = targetLoc
    const {start, end} = fnLoc

    const startOkay = start.line > sline || (start.line === sline && start.column >= scol)
    const endOkay = end.line < eline || (end.line === eline && end.column <= ecol)

    return startOkay && endOkay
}

module.exports = function(fileInfo, api, options) {
  const j = api.jscodeshift
    const {loc} = options

    const tokens = loc.split(',')
    const targetLoc = {
        sline: parseInt(tokens[0]),
        scol: parseInt(tokens[1]),
        eline: parseInt(tokens[2]),
        ecol: parseInt(tokens[3])
    }

  utils.checkValidSource(fileInfo, j)

  const result = j(fileInfo.source).findVariableDeclarators()
  const allIds = []
  result.forEach(path => {
      const varLoc = path.value.loc
      if (varLoc && isContained(varLoc, targetLoc)) {
          allIds.push(path.value.id.name)
      }
  })
  let currentSource = j(fileInfo.source)
  for (let id of allIds) {
    let name = utils.randomAlphanumericString(12)
    if (typeof id !== 'undefined') {
      currentSource.findVariableDeclarators(id).renameTo(name)
    }
  }
  return currentSource.toSource()
}