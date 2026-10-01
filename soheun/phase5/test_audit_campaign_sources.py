"""Source lookup must report missing and ambiguous candidates without choosing."""
import pathlib,tempfile
from audit_campaign_sources import audit,key


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp);(root/'MotherSamples').mkdir()
        def hp(seed):return {'signal_filename':'HH4b_picoAOD.h5','signal_ratio':0.,'seed':seed,'n_3b':1000000,'ratio_4b':.5}
        metadata={'one':hp(0),'absent':hp(2),'a':hp(3),'b':hp(3),'present':hp(4),'missing2':hp(4),'invalid':{}}
        for name in ('one','a','b','present'):(root/'MotherSamples'/name).touch()
        plan={'nodes':[{'id':str(i),'stage':1,'axes':{'signal':'HH4b','epsilon':'0','mother_seed':i}} for i in range(5)]}
        plan['nodes'].append({'stage':2})
        result=audit(plan,metadata,root)
        assert [r['status'] for r in result['cases']]==['UNIQUE_CANDIDATE','MISSING','MISSING_RECORD_FILE','AMBIGUOUS','AMBIGUOUS']
        assert result['source_cases']==5 and result['unindexed_metadata_records']==1
        assert key(hp(0))==key({**hp(0),'signal_ratio':'0.00'})
        assert key(hp(1))!=key(hp(1.5))
        assert not result['launch_submitted']
    print('PASS: exact source matching, no fractional-seed truncation, missing files and ambiguity reported')


if __name__=='__main__':main()
