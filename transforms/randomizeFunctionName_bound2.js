const utils = require('./lib/utils.js')

function contains(fnLoc, targetLoc) {
    const {sline, scol, eline, ecol} = targetLoc
    const {start, end} = fnLoc

    const startOkay = start.line < sline || (start.line === sline && start.column <= scol)
    const endOkay = end.line > eline || (end.line === eline && end.column >= ecol)

    return startOkay && endOkay
}

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

    let currentSource = j(fileInfo.source)
    const fnNames = {}

    // var func = null

    // find all fn declarations
    currentSource.find(j.FunctionDeclaration)
        .forEach(p => {
            const fnLoc = p.value.loc
            // const {start, end} = fnLoc
            // console.log(start.line, start.column, end.line, end.column)
            if (fnLoc && contains(fnLoc, targetLoc)) {
                const currentName = p.value.id.name
                fnNames[currentName] = utils.randomAlphanumericString(12)
                // func = p
                // console.log(2)
            }
        })

    // for all identifiers, if they match a mapping
    // change the name
    currentSource.find(j.Identifier)
        .forEach(p => {
            const idName = p.value.name
            // const fnLoc = p.value.loc
            // const {start, end} = fnLoc
            // console.log(start.line, start.column, end.line, end.column)
            // console.log(fnNames.hasOwnProperty(idName))
            if (fnNames.hasOwnProperty(idName)){  // && fnLoc && isContained(fnLoc, targetLoc)) {
                p.value.name = fnNames[idName]
            }
            return p
        })
    // const {start, end} = func.value.loc
    // console.log(start.line, start.column, end.line, end.column)
    return currentSource.toSource()
}