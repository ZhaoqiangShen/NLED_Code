function BASIS = get_basis(fsz)
    BASIS = kron(dctmtx(fsz), dctmtx(fsz));
    BASIS = BASIS(2:end, :)';
end